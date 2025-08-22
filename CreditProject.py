# credit_risk_model.py

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import sqlite3
import json
import warnings
from scipy.optimize import curve_fit, minimize_scalar, least_squares
from scipy.interpolate import interp1d
from scipy.stats import norm
from fredapi import Fred # For fetching economic data like Treasury yields
from typing import Optional, Callable, Tuple # Import Callable and Tuple for type hinting

# New imports for Bayesian Calibration
import pymc as pm
import arviz as az
import logging
import pytensor.tensor as pt # Explicitly import pytensor.tensor for symbolic operations
from pytensor.graph.op import Op # For custom PyTensor Op
from pytensor.graph.basic import Apply # Correct import for Apply node
from pytensor.tensor import TensorVariable # For type hinting in Op

# Suppress common warnings for cleaner output during development
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='fredapi') # Suppress FRED API key warning if not set

# Suppress PyMC/Theano/Aesara warnings
# Set logging levels to ERROR for libraries that might spam the console.
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("arviz").setLevel(logging.ERROR)
logging.getLogger("theano.tensor.rewriting").setLevel(logging.ERROR) # For PyMC3
logging.getLogger("aesara.tensor.rewriting").setLevel(logging.ERROR) # For PyMC (newer backend)
# Add other potential verbose loggers if they appear
logging.getLogger("numba").setLevel(logging.ERROR) # Numba can be verbose sometimes
logging.getLogger("pytensor").setLevel(logging.ERROR) # Suppress PyTensor warnings

# --- Configuration and Global Variables ---
# Application ID for potential Firebase integration (not directly used in this script's core logic)
appId = "credit-risk-model-app"
try:
    if '__app_id' in globals():
        appId = __app_id
except NameError:
    pass

# Firebase configuration (similarly, for potential external integration)
# Left empty as per "free" constraint unless specific external integration is requested later.
firebaseConfig = {}
try:
    if '__firebase_config' in globals():
        firebaseConfig = json.loads(__firebase_config)
except json.JSONDecodeError:
    firebaseConfig = {}


# Custom PyTensor Op to wrap the numerical Monte Carlo simulation
class CDSModelSpreadOp(Op):
    """
    A custom PyTensor Op that wraps the numerical Monte Carlo simulation for
    calculating CDS model spreads. This allows PyMC to treat the simulation
    as a symbolic node in its graph while executing the computation numerically.
    """
    def __init__(self, model_instance: 'CreditRiskModel'):
        # Pass the CreditRiskModel instance to the Op so it can access its methods
        self.model_instance = model_instance
        super().__init__()

    def make_node(self, kappa_l, sigma_l, lambda0, theta_maturities, theta_hazard_rates,
                  r0, kappa_r, theta_r, sigma_r, rho_correlation, recovery_rate, cds_maturities):
        # Define input and output types for PyTensor graph
        # pt.dscalar: double-precision scalar (float)
        # pt.dvector: double-precision vector (1D NumPy array)

        # Ensure all inputs are converted to PyTensor tensor variables with appropriate dtypes
        kappa_l = pt.as_tensor_variable(kappa_l, dtype='float64')
        sigma_l = pt.as_tensor_variable(sigma_l, dtype='float64')
        lambda0 = pt.as_tensor_variable(lambda0, dtype='float64')
        theta_maturities = pt.as_tensor_variable(theta_maturities, dtype='float64')
        theta_hazard_rates = pt.as_tensor_variable(theta_hazard_rates, dtype='float64')
        r0 = pt.as_tensor_variable(r0, dtype='float64')
        kappa_r = pt.as_tensor_variable(kappa_r, dtype='float64')
        theta_r = pt.as_tensor_variable(theta_r, dtype='float64')
        sigma_r = pt.as_tensor_variable(sigma_r, dtype='float64')
        rho_correlation = pt.as_tensor_variable(rho_correlation, dtype='float64')
        recovery_rate = pt.as_tensor_variable(recovery_rate, dtype='float64')
        cds_maturities = pt.as_tensor_variable(cds_maturities, dtype='float64')

        # The output of this Op is a vector of model spreads
        outputs = [pt.dvector()]

        # Create an Apply node that connects inputs to this Op and defines its outputs
        return Apply(self, [kappa_l, sigma_l, lambda0, theta_maturities, theta_hazard_rates,
                                     r0, kappa_r, theta_r, sigma_r, rho_correlation, recovery_rate, cds_maturities], outputs)

    def perform(self, node, inputs, outputs):
        """
        This method is called during the actual numerical computation within PyMC's sampling.
        Inputs here are concrete NumPy values (not symbolic PyTensor variables).
        """
        kappa_l_val, sigma_l_val, lambda0_fixed_val, theta_maturities_fixed_val, theta_hazard_rates_fixed_val, \
        r0_fixed_val, kappa_r_fixed_val, theta_r_fixed_val, sigma_r_fixed_val, rho_correlation_fixed_val, \
        rec_rate_fixed_val, cds_maturities_fixed_val = inputs

        # Reconstruct the theta_lambda_interp_func as a callable NumPy function
        # This interpolation function is based on fixed data derived from initial market spreads
        theta_func_reconstructed = interp1d(
            theta_maturities_fixed_val, theta_hazard_rates_fixed_val,
            kind='linear', fill_value="extrapolate", bounds_error=False
        )

        # Reconstruct the interest rate parameters dictionary
        ir_params_fixed_reconstructed = {
            'r0': r0_fixed_val.item(), # .item() extracts scalar from 0-d array
            'kappa_r': kappa_r_fixed_val.item(),
            'theta_r': theta_r_fixed_val.item(),
            'sigma_r': sigma_r_fixed_val.item(),
            'rho_correlation': rho_correlation_fixed_val.item()
        }

        # Call the original numerical simulation function from the CreditRiskModel instance.
        # All inputs here are now guaranteed to be numerical (NumPy scalars or arrays).
        model_spreads = self.model_instance._calculate_model_spreads_numerical(
            kappa_l=kappa_l_val.item(), # Extract scalar from 0-d array
            sigma_l=sigma_l_val.item(), # Extract scalar from 0-d array
            maturities=cds_maturities_fixed_val,
            lambda0_fixed=lambda0_fixed_val.item(),
            theta_func_fixed=theta_func_reconstructed,
            ir_params_fixed=ir_params_fixed_reconstructed,
            rec_rate_fixed=rec_rate_fixed_val.item(),
            num_mc_paths=1000, # Use moderate paths for MCMC performance
            num_mc_steps=50
        )

        # The result must be placed into the first output buffer (outputs[0][0])
        outputs[0][0] = np.asarray(model_spreads)


class CreditRiskModel:
    """
    A comprehensive model for credit risk analysis, Credit Default Swap (CDS) pricing,
    and risky bond valuation, incorporating a stochastic default intensity model
    (CIR++ type) and interest rate dynamics.

    This class integrates:
    1.  Yield Curve Modeling (Nelson-Siegel-Svensson): Fitting a smooth, parametric yield curve
        from observed U.S. Treasury data (fetched from FRED).
    2.  Stochastic Default Intensity: Implementation of a Cox-Ingersoll-Ross (CIR)
        type stochastic process for the hazard rate, extended (CIR++) to fit an
        initial credit (CDS) spread curve.
    3.  Credit Default Swap (CDS) Pricing: Monte Carlo simulation for valuing CDS
        contracts, accounting for both stochastic interest rates and stochastic
        default intensity.
    4.  Risky Bond Valuation: Pricing bonds incorporating the derived default
        probabilities and credit-adjusted discounting.
    5.  SQLite Database Integration: Persistent storage and retrieval of market
        data (yields, simulated CDS spreads) and model outputs (calibrated parameters).
    6.  Visualization: Plots for yield curves, simulated paths, and calibration results.
    7.  Bayesian Calibration (New): Utilizes Markov Chain Monte Carlo (MCMC)
        to infer the posterior distributions of default intensity parameters.

    Key Features:
    - Fetches real-world historical U.S. Treasury yield data (FRED API).
    - Implements a robust bootstrapping algorithm for constructing the zero-coupon curve.
    - Uses SQLite for local, file-based data persistence.
    - Models default intensity using a mean-reverting square-root process (CIR type).
    - Prices CDS contracts using a two-factor Monte Carlo simulation (interest rate + default intensity).
    - Calibrates the stochastic intensity model parameters to synthetic market CDS spreads
      using both traditional least-squares and advanced Bayesian MCMC.
    - Provides valuation for bonds subject to credit risk.
    """

    def __init__(self, fred_api_key: Optional[str] = None, db_name: str = 'credit_risk_data.db'):

        # Initialize connection and cursor to None immediately to prevent AttributeError if Fred() fails
        self.conn = None
        self.cursor = None

        self.fred = Fred(api_key=fred_api_key)
        self.db_name = db_name


        # --- Market Data Storage ---
        self.yield_curve_data: Optional[pd.DataFrame] = None # Stores bootstrapped zero rates and discount factors
        self.market_bond_data: Optional[pd.DataFrame] = None # Stores market data for bonds used in bootstrapping
        self.simulated_cds_market_data: Optional[pd.DataFrame] = None # Stores synthetic CDS market spreads

        # --- Model Parameters ---
        # Risk-free interest rate model (e.g., flat rate or simple Vasicek for MC baseline)
        self.r0 = 0.015 # Initial short rate for interest rate process in MC
        self.kappa_r = 0.1 # Mean reversion speed for short rate (example, can be calibrated)
        self.theta_r = 0.02 # Long-term mean for short rate
        self.sigma_r = 0.01 # Volatility for short rate

        # Nelson-Siegel-Svensson (NSS) Yield Curve Parameters
        # beta_0, beta_1, beta_2, beta_3, lambda_1, lambda_2
        # These will be calibrated to market Treasury yields
        self.nss_params: Optional[dict[str, float]] = None


        # Stochastic Default Intensity (CIR++) Model Parameters for lambda_t
        # d_lambda_t = kappa_lambda * (theta_lambda(t) - lambda_t) dt + sigma_lambda * sqrt(lambda_t) dW_t^lambda
        self.kappa_lambda: Optional[float] = None # Speed of mean reversion for intensity
        self.sigma_lambda: Optional[float] = None # Volatility of intensity
        self.theta_lambda_func: Optional[Callable] = None # Time-dependent mean intensity (calibrated to CDS curve)
        self.lambda0: Optional[float] = None # Initial default intensity (can be set from first CDS spread)

        self.rho_correlation: Optional[float] = 0.0 # Correlation between interest rate and default intensity Wiener processes

        self.recovery_rate = 0.4 # Default recovery rate (40% is common for corporate bonds)


        # Initialize database connection and tables
        self.connect_db()
        self.create_tables()

    def __del__(self):
        """Ensures the database connection is closed when the object is destroyed."""
        if self.conn:
            self.conn.close()
            print(f"Database connection to {self.db_name} closed.")

    # --- SQLite Database Methods ---
    def connect_db(self):
        """Establishes a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            print(f"Connected to SQLite database: {self.db_name}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None
            self.cursor = None

    def create_tables(self):
        """Creates necessary tables in the SQLite database."""
        if not self.conn:
            print("Cannot create tables: No database connection.")
            return

        try:
            # Table for Raw Treasury Data (e.g., from FRED)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS treasury_yields (
                    date TEXT PRIMARY KEY,
                    yield_3m REAL,
                    yield_1y REAL,
                    yield_2y REAL,
                    yield_3y REAL,
                    yield_5y REAL,
                    yield_7y REAL,
                    yield_10y REAL,
                    yield_20y REAL,
                    yield_30y REAL
                );
            ''')

            # Table for Zero-Coupon Yield Curve (NSS parameters)
            # Store NSS parameters directly rather than bootstrapped points
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS nss_yield_curve_params (
                    calibration_date TEXT PRIMARY KEY,
                    beta0 REAL,
                    beta1 REAL,
                    beta2 REAL,
                    beta3 REAL,
                    lambda1 REAL,
                    lambda2 REAL
                );
            ''')

            # Table for Simulated CDS Market Data
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS simulated_cds_spreads (
                    maturity REAL PRIMARY KEY, -- Maturity of the CDS in years
                    spread REAL                -- Simulated market spread in bps
                );
            ''')

            # Table for Calibrated Model Parameters (e.g., for CIR++)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_parameters (
                    model_name TEXT,
                    parameter_name TEXT,
                    value REAL,
                    calibration_date TEXT,
                    PRIMARY KEY (model_name, parameter_name)
                );
            ''')

            # Table for CDS Pricing Results
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS cds_pricing_results (
                    cds_id TEXT PRIMARY KEY,
                    notional REAL,
                    maturity REAL,
                    coupon REAL,
                    recovery_rate REAL,
                    pricing_date TEXT,
                    model_price REAL,
                    model_type TEXT,
                    calibrated_intensity_params TEXT, -- Store as JSON string
                    calibrated_ir_params TEXT         -- Store as JSON string
                );
            ''')

            self.conn.commit()
            print("Database tables checked/created successfully.")
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")

    def insert_treasury_yields(self, df: pd.DataFrame):
        """
        Inserts fetched FRED Treasury yield data into the treasury_yields table.

        Args:
            df (pd.DataFrame): DataFrame containing yield data with 'date' as index.
        """
        if not self.conn:
            print("Cannot insert data: No database connection.")
            return

        # Rename FRED series IDs to more friendly column names for the DB
        column_mapping = {
            'DGS3MO': 'yield_3m', 'DGS1': 'yield_1y', 'DGS2': 'yield_2y',
            'DGS3': 'yield_3y', 'DGS5': 'yield_5y', 'DGS7': 'yield_7y',
            'DGS10': 'yield_10y', 'DGS20': 'yield_20y', 'DGS30': 'yield_30y'
        }
        df_to_insert = df.rename(columns=column_mapping)

        # Convert index (date) to string format for SQLite
        df_to_insert['date'] = df_to_insert.index.strftime('%Y-%m-%d')
        df_to_insert = df_to_insert.reset_index(drop=True)

        # Ensure all expected columns are present, fill missing with NaN (which SQLite converts to NULL)
        expected_columns = ['date', 'yield_3m', 'yield_1y', 'yield_2y', 'yield_3y',
                            'yield_5y', 'yield_7y', 'yield_10y', 'yield_20y', 'yield_30y']
        for col in expected_columns:
            if col not in df_to_insert.columns:
                df_to_insert[col] = np.nan

        # Select and reorder columns to match the table schema
        df_to_insert = df_to_insert[expected_columns]

        try:
            # Use executemany for efficient insertion
            self.cursor.executemany(
                '''
                INSERT OR REPLACE INTO treasury_yields (date, yield_3m, yield_1y, yield_2y, yield_3y, yield_5y, yield_7y, yield_10y, yield_20y, yield_30y)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                ''',
                df_to_insert.values.tolist() # Convert DataFrame rows to a list of tuples
            )
            self.conn.commit()
            print(f"Successfully inserted/updated {len(df_to_insert)} rows in treasury_yields table.")
        except sqlite3.Error as e:
            print(f"Error inserting treasury yields: {e}")

    def get_treasury_yields(self, latest_n_days: int = 1) -> Optional[pd.DataFrame]:
        """
        Retrieves the latest available Treasury yield data from the database.

        Args:
            latest_n_days (int): Number of latest days to retrieve. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame with the latest Treasury yields, or None if no data.
        """
        if not self.conn:
            print("Cannot retrieve data: No database connection.")
            return None
        try:
            query = f"SELECT * FROM treasury_yields ORDER BY date DESC LIMIT {latest_n_days};"
            df = pd.read_sql_query(query, self.conn, index_col='date', parse_dates=['date'])
            if not df.empty:
                print(f"Retrieved {len(df)} latest Treasury yield entries.")
                return df
            else:
                print("No Treasury yield data found in the database.")
                return None
        except pd.io.sql.DatabaseError as e:
            print(f"Error retrieving treasury yields: {e}")
            return None

    def insert_nss_params(self, params: dict):
        """
        Inserts fitted NSS parameters into the database.

        Args:
            params (dict): Dictionary of NSS parameters (beta0, beta1, beta2, beta3, lambda1, lambda2).
        """
        if not self.conn:
            print("Cannot insert NSS parameters: No database connection.")
            return

        calibration_date = datetime.date.today().strftime('%Y-%m-%d')
        try:
            self.cursor.execute('DELETE FROM nss_yield_curve_params;') # Clear existing NSS params
            self.conn.commit()
            self.cursor.execute(
                '''
                INSERT INTO nss_yield_curve_params (calibration_date, beta0, beta1, beta2, beta3, lambda1, lambda2)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                ''',
                (calibration_date, params['beta0'], params['beta1'], params['beta2'],
                 params['beta3'], params['lambda1'], params['lambda2'])
            )
            self.conn.commit()
            print(f"Successfully inserted NSS parameters for {calibration_date}.")
        except sqlite3.Error as e:
            print(f"Error inserting NSS parameters: {e}")

    def get_nss_params(self) -> Optional[dict]:
        """
        Retrieves the latest NSS parameters from the database.

        Returns:
            dict: Dictionary of NSS parameters, or None if not found.
        """
        if not self.conn:
            print("Cannot retrieve NSS parameters: No database connection.")
            return None
        try:
            self.cursor.execute("SELECT * FROM nss_yield_curve_params ORDER BY calibration_date DESC LIMIT 1;")
            row = self.cursor.fetchone()
            if row:
                columns = [description[0] for description in self.cursor.description]
                params = dict(zip(columns, row))
                # Remove calibration_date from the returned dictionary if not needed for direct curve calculation
                params.pop('calibration_date', None)
                print("Retrieved NSS parameters from DB.")
                self.nss_params = params # Set the model's attribute
                return params
            else:
                print("No NSS parameters found in the database.")
                return None
        except sqlite3.Error as e:
            print(f"Error retrieving NSS parameters: {e}")
            return None

    def insert_simulated_cds_market_data(self, df: pd.DataFrame):
        """
        Inserts simulated CDS market data into the database.

        Args:
            df (pd.DataFrame): DataFrame with 'maturity' and 'spread' columns.
        """
        if not self.conn:
            print("Cannot insert data: No database connection.")
            return
        try:
            self.cursor.execute('DELETE FROM simulated_cds_spreads;') # Clear existing data
            self.conn.commit()
            self.cursor.executemany(
                '''
                INSERT INTO simulated_cds_spreads (maturity, spread)
                VALUES (?, ?);
                ''',
                df[['maturity', 'spread']].values.tolist()
            )
            self.conn.commit()
            print(f"Successfully inserted {len(df)} entries into simulated_cds_spreads table.")
        except sqlite3.Error as e:
            print(f"Error inserting simulated CDS market data: {e}")

    def get_simulated_cds_market_data(self) -> Optional[pd.DataFrame]:
        """
        Retrieves simulated CDS market data from the database.

        Returns:
            pd.DataFrame: DataFrame with 'maturity' and 'spread', or None.
        """
        if not self.conn:
            print("Cannot retrieve data: No database connection.")
            return None
        try:
            df = pd.read_sql_query("SELECT * FROM simulated_cds_spreads ORDER BY maturity;", self.conn)
            if not df.empty:
                print(f"Retrieved simulated CDS market data with {len(df)} points.")
                self.simulated_cds_market_data = df
                return df
            else:
                print("No simulated CDS market data found in the database.")
                return None
        except pd.io.sql.DatabaseError as e:
            print(f"Error retrieving simulated CDS market data: {e}")
            return None

    def insert_model_parameters(self, model_name: str, params: dict):
        """
        Inserts or updates model parameters in the database.

        Args:
            model_name (str): Name of the model (e.g., 'CIR++_Intensity').
            params (dict): Dictionary of parameter names and their values.
        """
        if not self.conn:
            print("Cannot insert parameters: No database connection.")
            return

        calibration_date = datetime.date.today().strftime('%Y-%m-%d')
        try:
            for param_name, value in params.items():
                self.cursor.execute(


                    (model_name, param_name, value, calibration_date)
                )
            self.conn.commit()
            print(f"Successfully inserted/updated parameters for model '{model_name}'.")
        except sqlite3.Error as e:
            print(f"Error inserting model parameters: {e}")

    def get_model_parameters(self, model_name: str) -> dict:
        """
        Retrieves model parameters from the database.

        Args:
            model_name (str): Name of the model to retrieve parameters for.

        Returns:
            dict: Dictionary of parameter names and their values, or empty dict if not found.
        """
        if not self.conn:
            print("Cannot retrieve parameters: No database connection.")
            return {}
        try:
            self.cursor.execute(
                "SELECT parameter_name, value FROM model_parameters WHERE model_name = ?;",
                (model_name,)
            )
            rows = self.cursor.fetchall()
            params = {row[0]: row[1] for row in rows}
            if params:
                print(f"Retrieved parameters for model '{model_name}'.")
            else:
                print(f"No parameters found for model '{model_name}'.")
            return params
        except sqlite3.Error as e:
            print(f"Error retrieving model parameters: {e}")
            return {}

    def insert_cds_pricing_result(self, cds_id: str, notional: float, maturity: float, coupon: float,
                                  recovery_rate: float, model_price: float, model_type: str,
                                  calibrated_intensity_params: dict, calibrated_ir_params: dict):
        """
        Inserts a CDS pricing result into the database.

        Args:
            cds_id (str): Unique ID for the CDS.
            notional (float): Notional amount of the CDS.
            maturity (float): Maturity of the CDS.
            coupon (float): Annual coupon of the CDS.
            recovery_rate (float): Recovery rate in case of default.
            model_price (float): The calculated model price of the CDS.
            model_type (str): The model used for pricing (e.g., 'CIR++ MC').
            calibrated_intensity_params (dict): Dictionary of calibrated intensity model parameters.
            calibrated_ir_params (dict): Dictionary of calibrated interest rate model parameters.
        """
        if not self.conn:
            print("Cannot insert CDS pricing result: No database connection.")
            return

        pricing_date = datetime.date.today().strftime('%Y-%m-%d')
        try:
            self.cursor.execute(

                (
                    cds_id, notional, maturity, coupon, recovery_rate, pricing_date,
                    model_price, model_type, json.dumps(calibrated_intensity_params),
                    json.dumps(calibrated_ir_params)
                )
            )
            self.conn.commit()
            print(f"Successfully inserted/updated CDS pricing result for CDS ID: {cds_id}.")
        except sqlite3.Error as e:
            print(f"Error inserting CDS pricing result: {e}")

    def get_cds_pricing_results(self) -> Optional[pd.DataFrame]:
        """
        Retrieves all CDS pricing results from the database.

        Returns:
            pd.DataFrame: DataFrame containing CDS pricing results, or None.
        """
        if not self.conn:
            print("Cannot retrieve CDS pricing results: No database connection.")
            return None
        try:
            df = pd.read_sql_query("SELECT * FROM cds_pricing_results ORDER BY pricing_date DESC, maturity ASC;", self.conn)
            if not df.empty:
                # Convert JSON strings back to dicts
                df['calibrated_intensity_params'] = df['calibrated_intensity_params'].apply(json.loads)
                df['calibrated_ir_params'] = df['calibrated_ir_params'].apply(json.loads)
                print(f"Retrieved {len(df)} CDS pricing results.")
                return df
            else:
                print("No CDS pricing results found in the database.")
                return None
        except pd.io.sql.DatabaseError as e:
            print(f"Error retrieving CDS pricing results: {e}")
            return None


    # --- Yield Curve Modeling Methods (Nelson-Siegel-Svensson) ---
    def fetch_treasury_yields(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical U.S. Treasury Par Yields from FRED.
        These yields will be used as inputs for NSS fitting.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame with fetched yields for the latest available date
                          that has at least one valid yield, or None if failed.
        """
        print(f"\nFetching U.S. Treasury Par Yields from FRED from {start_date} to {end_date}...")
        # FRED series IDs for various maturities (DGS = Daily Treasury Yield Curve Rates)
        series_ids = {
            'DGS3MO': '3-Month', 'DGS1': '1-Year', 'DGS2': '2-Year', 'DGS3': '3-Year',
            'DGS5': '5-Year', 'DGS7': '7-Year', 'DGS10': '10-Year', 'DGS20': '20-Year', 'DGS30': '30-Year'
        }

        try:
            # Fetch data for all series
            yields_data = {}
            for series_id, maturity_label in series_ids.items():
                df = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                yields_data[series_id] = df.rename(f'{maturity_label} Yield')

            # Combine into a single DataFrame
            combined_df = pd.DataFrame(yields_data)
            print(f"\nRaw combined_df from FRED (head):\n{combined_df.head()}")
            print(f"\nRaw combined_df from FRED (tail):\n{combined_df.tail()}")

            # Store the raw fetched data into the database before renaming columns for internal use
            # This part needs the original DGS names for insert_treasury_yields
            original_combined_df_for_db = pd.DataFrame(yields_data).dropna(how='all')
            original_numeric_cols_for_db = original_combined_df_for_db.select_dtypes(include=np.number).columns
            original_combined_df_for_db[original_numeric_cols_for_db] = original_combined_df_for_db[original_numeric_cols_for_db] / 100.0
            self.insert_treasury_yields(original_combined_df_for_db)

            # Rename FRED series IDs to more friendly column names for internal processing and filtering
            # This mapping is crucial before filtering for 'Yield' columns
            reverse_column_mapping = {
                'DGS3MO': '3-Month Yield', 'DGS1': '1-Year Yield', 'DGS2': '2-Year Yield',
                'DGS3': '3-Year Yield', 'DGS5': '5-Year Yield', 'DGS7': '7-Year Yield',
                'DGS10': '10-Year Yield', 'DGS20': '20-Year Yield', 'DGS30': '30-Year Yield'
            }
            combined_df = combined_df.rename(columns=reverse_column_mapping)
            print(f"\nCombined_df after renaming columns (head):\n{combined_df.head()}")

            # Filter out rows where all values are NaN before processing
            combined_df = combined_df.dropna(how='all')

            if combined_df.empty:
                print("No Treasury yield data fetched from FRED after dropping all-NaN rows. Please check dates or FRED API key.")
                return None

            # Convert yields from percentage to decimal (e.g., 2.5% -> 0.025)
            # Apply to the numeric values, leaving NaNs as is
            numeric_cols = combined_df.select_dtypes(include=np.number).columns
            combined_df[numeric_cols] = combined_df[numeric_cols] / 100.0

            # Find the latest date with at least one non-NaN yield
            latest_date_data = None
            # Iterate backwards from the most recent date to find the first valid row
            for idx in range(len(combined_df) - 1, -1, -1):
                row_data = combined_df.iloc[idx]
                # Now, row_data.filter(regex='Yield') will correctly find columns like '3-Month Yield'
                if not row_data.filter(regex='Yield').dropna().empty:
                    latest_date_data = row_data
                    break

            if latest_date_data is None:
                print("No Treasury yield data with actual values found for any date in the fetched range after conversion.")
                return None

            latest_date_df = pd.DataFrame(latest_date_data).T # Convert Series to DataFrame (single row)
            latest_date_df.index = [latest_date_data.name] # Restore original date index
            latest_date_df.index.name = 'date'

            print(f"\nLatest valid Treasury yields for bootstrapping:\n{latest_date_df}") # Diagnostic print

            self.market_bond_data = latest_date_df # Store only the single latest non-null row for bootstrapping

            print(f"Successfully fetched Treasury yields for {latest_date_df.index[0].strftime('%Y-%m-%d')}.")
            return latest_date_df

        except Exception as e:
            print(f"Error fetching Treasury yields from FRED: {e}")
            return None

    def _get_maturity_in_years(self, label: str) -> float:
        """Helper to convert maturity labels (e.g., '3-Month', '10-Year') to years."""
        if 'Month' in label:
            return float(label.split('-')[0]) / 12.0
        elif 'Year' in label:
            return float(label.split('-')[0])
        return np.nan # Should not happen with FRED series

    @staticmethod
    def _nss_zero_rate_func(t: np.ndarray, beta0: float, beta1: float, beta2: float, beta3: float, lambda1: float, lambda2: float) -> np.ndarray:
        """
        Calculates the Nelson-Siegel-Svensson (NSS) zero-coupon rate for a given maturity t.

        Args:
            t (np.ndarray): Time to maturity in years (can be a scalar or array).
            beta0, beta1, beta2, beta3 (float): NSS parameters.
            lambda1, lambda2 (float): Decay parameters (must be > 0).

        Returns:
            np.ndarray: NSS zero-coupon rate at maturity t.
        """
        # Handle potential division by zero if lambda is extremely small
        lambda1 = np.maximum(lambda1, 1e-8)
        lambda2 = np.maximum(lambda2, 1e-8)

        term1 = beta0
        term2 = beta1 * (1 - np.exp(-t / lambda1)) / (t / lambda1)
        term3 = beta2 * ((1 - np.exp(-t / lambda1)) / (t / lambda1) - np.exp(-t / lambda1))
        term4 = beta3 * ((1 - np.exp(-t / lambda2)) / (t / lambda2) - np.exp(-t / lambda2))


        if isinstance(t, np.ndarray):
            result = np.zeros_like(t, dtype=float)
            # Find where t is close to zero
            t_is_zero = np.isclose(t, 0.0)

            # Apply the formula for non-zero t
            t_nonzero = t[~t_is_zero]
            result[~t_is_zero] = (term1 +
                                  beta1 * (1 - np.exp(-t_nonzero / lambda1)) / (t_nonzero / lambda1) +
                                  beta2 * ((1 - np.exp(-t_nonzero / lambda1)) / (t_nonzero / lambda1) - np.exp(-t_nonzero / lambda1)) +
                                  beta3 * ((1 - np.exp(-t_nonzero / lambda2)) / (t_nonzero / lambda2) - np.exp(-t_nonzero / lambda2)))

            # Apply the t=0 rule for t close to zero
            result[t_is_zero] = beta0 + beta1
            return result
        else: # Scalar input
            if np.isclose(t, 0.0):
                return beta0 + beta1
            else:
                return term1 + term2 + term3 + term4


    def _nss_par_yield_func(self, t: np.ndarray, beta0: float, beta1: float, beta2: float, beta3: float, lambda1: float, lambda2: float) -> np.ndarray:
        """
        Calculates the par yield implied by the NSS zero-coupon curve.
        For a bond priced at par, its coupon rate equals its yield-to-maturity.
        Par yield = (1 - DF(T)) / Sum(DF(ti)) for coupon payments + DF(T) for principal.
        Assuming annual coupon payments for simplicity for par yield calculation here.

        Args:
            t (np.ndarray): Time to maturity in years (can be a scalar or array).
            beta0, beta1, beta2, beta3 (float): NSS parameters.
            lambda1, lambda2 (float): Decay parameters.

        Returns:
            np.ndarray: NSS implied par yield at maturity t.
        """
        if isinstance(t, np.ndarray):
            par_yields = np.zeros_like(t, dtype=float)
            for i, maturity in enumerate(t):
                if np.isclose(maturity, 0.0):
                    # For T=0, par yield is just the instantaneous rate
                    par_yields[i] = self._nss_zero_rate_func(0.0, beta0, beta1, beta2, beta3, lambda1, lambda2)
                    continue

                # Calculate discount factors for coupon payments (assuming annual)
                coupon_payment_times = np.arange(1.0, maturity + 1e-9, 1.0)
                # Ensure coupon payment times don't exceed maturity
                coupon_payment_times = coupon_payment_times[coupon_payment_times <= maturity]

                # Get zero rates for each coupon payment time
                zero_rates_at_coupon_times = self._nss_zero_rate_func(coupon_payment_times, beta0, beta1, beta2, beta3, lambda1, lambda2)
                discount_factors_at_coupon_times = np.exp(-zero_rates_at_coupon_times * coupon_payment_times)

                # Discount factor at maturity
                df_maturity = np.exp(-self._nss_zero_rate_func(maturity, beta0, beta1, beta2, beta3, lambda1, lambda2) * maturity)

                sum_dfs = np.sum(discount_factors_at_coupon_times)

                # Handle edge case where sum_dfs is very small or zero
                if sum_dfs + df_maturity > 1e-12: # Avoid division by zero
                    par_yields[i] = (1.0 - df_maturity) / sum_dfs
                else:
                    par_yields[i] = 1.0 # Set to a large value to penalize bad fits

            return par_yields
        else: # Scalar input
            maturity = t
            if np.isclose(maturity, 0.0):
                return self._nss_zero_rate_func(0.0, beta0, beta1, beta2, beta3, lambda1, lambda2)

            coupon_payment_times = np.arange(1.0, maturity + 1e-9, 1.0)
            coupon_payment_times = coupon_payment_times[coupon_payment_times <= maturity]

            zero_rates_at_coupon_times = self._nss_zero_rate_func(coupon_payment_times, beta0, beta1, beta2, beta3, lambda1, lambda2)
            discount_factors_at_coupon_times = np.exp(-zero_rates_at_coupon_times * coupon_payment_times)

            df_maturity = np.exp(-self._nss_zero_rate_func(maturity, beta0, beta1, beta2, beta3, lambda1, lambda2) * maturity)

            sum_dfs = np.sum(discount_factors_at_coupon_times)

            if sum_dfs + df_maturity > 1e-12:
                return (1.0 - df_maturity) / sum_dfs
            else:
                return 1.0 # Penalize bad fit

    def fit_nss_yield_curve(self, market_data_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Fits the Nelson-Siegel-Svensson (NSS) yield curve model to market observed par yields.

        Args:
            market_data_df (pd.DataFrame): A DataFrame (typically a single row)
                                          containing par yields for various maturities.
                                          Expected columns: '3-Month Yield', '1-Year Yield', ..., '30-Year Yield'.

        Returns:
            pd.DataFrame: A DataFrame with 'maturity', 'zero_rate', 'discount_factor' columns
                          representing the fitted NSS curve, or None if fitting fails.
        """
        print("\nStarting Nelson-Siegel-Svensson (NSS) yield curve fitting...")
        if market_data_df.empty:
            print("No market data provided for NSS fitting.")
            return None

        # Extract relevant market data (maturities and par yields)
        latest_market_yields = market_data_df.iloc[0].filter(regex='Yield').dropna()
        if latest_market_yields.empty:
            print("No valid market yields found for NSS fitting. Cannot fit yield curve.")
            return None

        maturities = np.array([self._get_maturity_in_years(col.replace(' Yield', '')) for col in latest_market_yields.index])
        observed_yields = latest_market_yields.values

        # Sort by maturity
        sorted_indices = np.argsort(maturities)
        maturities = maturities[sorted_indices]
        observed_yields = observed_yields[sorted_indices]

        # Define the objective function for least squares (residuals)
        # params order: [beta0, beta1, beta2, beta3, lambda1, lambda2]
        def residuals(params: np.ndarray, t_values: np.ndarray, market_yields: np.ndarray) -> np.ndarray:
            b0, b1, b2, b3, l1, l2 = params

            # Penalize invalid lambda values heavily to guide optimizer
            if l1 <= 0 or l2 <= 0:
                return np.full_like(market_yields, 1e6) # Large error

            # Calculate model-implied par yields
            model_yields = self._nss_par_yield_func(t_values, b0, b1, b2, b3, l1, l2)

            # Handle NaNs or infinities that might result from numerical instabilities
            model_yields = np.nan_to_num(model_yields, nan=1e6, posinf=1e6, neginf=1e6)

            # Return the difference (residuals)
            return market_yields - model_yields

        # Initial guess for NSS parameters
        # These are common starting points; can be refined.
        # beta0 (long-term level), beta1 (slope), beta2 (curvature), beta3 (second hump)
        # lambda1 (short-to-medium decay), lambda2 (medium-to-long decay)
        initial_guess = [
            observed_yields[-1],  # beta0: long end of the curve
            observed_yields[0] - observed_yields[-1], # beta1: short minus long (slope)
            0.0, # beta2: curvature, often starts near zero
            0.0, # beta3: second hump, often starts near zero
            1.0, # lambda1: decay factor for short-to-medium term
            10.0 # lambda2: decay factor for medium-to-long term
        ]

        # Bounds for parameters:
        # beta0, beta1, beta2, beta3 can be negative (negative rates are possible)
        # lambda1, lambda2 must be positive
        bounds_lower = [-0.5, -0.5, -0.5, -0.5, 1e-3, 1e-3]
        bounds_upper = [0.5, 0.5, 0.5, 0.5, 5.0, 30.0] # Lambda values typically 0.1 to 20ish

        try:
            # Use least_squares to fit the model
            # `x_scale='jac'` often helps for better scaling across parameters
            result = least_squares(
                residuals,
                initial_guess,
                bounds=(bounds_lower, bounds_upper),
                args=(maturities, observed_yields),
                method='trf', # Trust-Region-Reflective algorithm
                ftol=1e-8, xtol=1e-8, gtol=1e-8, # Increase tolerance for precision
                max_nfev=1000 # Max function evaluations
            )

            if result.success:
                b0, b1, b2, b3, l1, l2 = result.x
                self.nss_params = {
                    'beta0': b0, 'beta1': b1, 'beta2': b2, 'beta3': b3,
                    'lambda1': l1, 'lambda2': l2
                }
                self.insert_nss_params(self.nss_params)
                print(f"NSS yield curve fitting successful! Calibrated parameters: {self.nss_params}")

                # Generate a fine grid of maturities for the smooth curve
                fine_maturities = np.linspace(maturities.min(), maturities.max() + 5, 100) # Extend a bit beyond max observed
                if maturities.min() > 0: # Ensure we start from 0 if relevant
                    fine_maturities = np.linspace(0.0, maturities.max() + 5, 100)

                fitted_zero_rates = self._nss_zero_rate_func(fine_maturities, **self.nss_params)
                fitted_discount_factors = np.exp(-fitted_zero_rates * fine_maturities)

                # Ensure discount factors are capped at 1.0 and positive
                fitted_discount_factors = np.clip(fitted_discount_factors, 1e-8, 1.0)

                fitted_curve_df = pd.DataFrame({
                    'maturity': fine_maturities,
                    'zero_rate': fitted_zero_rates,
                    'discount_factor': fitted_discount_factors
                })
                self.yield_curve_data = fitted_curve_df # Update the internal yield_curve_data

                return fitted_curve_df
            else:
                print(f"NSS yield curve fitting failed: {result.message}")
                self.nss_params = None
                self.yield_curve_data = None
                return None
        except Exception as e:
            print(f"An error occurred during NSS yield curve fitting: {e}")
            self.nss_params = None
            self.yield_curve_data = None
            return None


    def get_discount_factor(self, t_in_years: float) -> float:
        """
        Retrieves the discount factor for a given time to maturity (t) from the
        fitted NSS yield curve.

        Args:
            t_in_years (float): Time to maturity in years.

        Returns:
            float: The NSS implied discount factor. Returns NaN if NSS parameters are not available
                   or if t is negative.
        """
        if self.nss_params is None:
            # Try to load from DB if not already loaded
            self.nss_params = self.get_nss_params()
            if self.nss_params is None:
                print("Error: NSS parameters not available or not fitted.")
                return np.nan

        if t_in_years < 0:
            return np.nan

        # Calculate zero rate using NSS function
        zero_rate = self._nss_zero_rate_func(t_in_years, **self.nss_params)

        # Calculate discount factor
        df = np.exp(-zero_rate * t_in_years)
        return np.clip(df, 1e-8, 1.0) # Ensure discount factor is between 0 and 1

    def get_zero_rate(self, t_in_years: float) -> float:
        """
        Retrieves the zero-coupon rate for a given time to maturity (t) from the
        fitted NSS yield curve.

        Args:
            t_in_years (float): Time to maturity in years.

        Returns:
            float: The NSS implied zero-coupon rate. Returns NaN if NSS parameters are not available
                   or if t is negative.
        """
        if self.nss_params is None:
            # Try to load from DB if not already loaded
            self.nss_params = self.get_nss_params()
            if self.nss_params is None:
                print("Error: NSS parameters not available or not fitted.")
                return np.nan

        if t_in_years < 0:
            return np.nan

        # Calculate zero rate directly using NSS function
        rate = self._nss_zero_rate_func(t_in_years, **self.nss_params)
        return np.maximum(rate, -0.05) # Cap at -5% to avoid extreme negative rates if model extrapolates poorly


    # --- Stochastic Default Intensity Model (CIR++ Type) ---
    def _lambda_cir_plusplus(self, t: float, lambda0: float, kappa_lambda: float, sigma_lambda: float,
                             theta_lambda_interp_func: Callable, W_lambda_t: float) -> float:
        """
        Calculates the instantaneous default intensity (lambda_t) based on a CIR++ like process.
        This is not a full SDE solver, but rather calculates a single step given the Wiener process increment.
        The `theta_lambda_interp_func` allows for a time-dependent long-term mean to fit the initial curve.

        Args:
            t (float): Current time in years.
            lambda0 (float): Initial intensity.
            kappa_lambda (float): Speed of mean reversion for intensity.
            sigma_lambda (float): Volatility of intensity.
            theta_lambda_interp_func (Callable): Interpolation function for theta(t).
            W_lambda_t (float): Wiener process increment for this step.

        Returns:
            float: The instantaneous default intensity at time t.
        """
        # This function would typically be used inside a Monte Carlo loop to evolve lambda_t.
        # For a single point, we just need the theta_t value at time t.
        theta_t_val = theta_lambda_interp_func(t) if theta_lambda_interp_func else self.lambda0 # Fallback if func not set

        # In a real MC, this is the previous lambda_t, not lambda0.
        # This is a conceptual function; the actual path evolution is in `_simulate_paths`.
        return np.maximum(lambda0, 1e-8) # Placeholder for now, actual evolution happens in MC.

    def _theta_lambda_calibration_target(self, t_values: np.ndarray, initial_cds_spreads: np.ndarray,
                                        cds_maturities: np.ndarray, r0: float) -> Tuple[Callable, np.ndarray, np.ndarray]:
        """
        Derives the time-dependent theta(t) function for the CIR++ default intensity model.
        This theta(t) ensures the model fits the initial term structure of CDS spreads.


        Args:
            t_values (np.ndarray): Array of time points for which to define theta(t).
            initial_cds_spreads (np.ndarray): Array of market CDS spreads (decimal) at `cds_maturities`.
            cds_maturities (np.ndarray): Array of CDS maturities in years.
            r0 (float): The initial short rate for basic discounting approximation in this derivation.

        Returns:
            Tuple[Callable, np.ndarray, np.ndarray]:
                - An interpolation function `theta_lambda_interp_func(t)` that returns theta(t).
                - The `cds_maturities` (x-values for interpolation).
                - The `hazard_rates` (y-values for interpolation).
        """
        print("\nDeriving time-dependent theta_lambda(t) for CIR++ model...")

        if len(initial_cds_spreads) < 2:
            print("Not enough CDS spreads to derive theta(t). Returning constant theta based on first spread.")
            # If only one spread, assume constant hazard rate implied by that spread
            avg_spread = np.mean(initial_cds_spreads) if len(initial_cds_spreads) > 0 else 0.01 # Default to 1%
            # Simple approximation: hazard rate ~ spread / (1 - recovery_rate)
            const_hazard = avg_spread / (1 - self.recovery_rate) if (1 - self.recovery_rate) > 1e-8 else avg_spread
            return (lambda t: const_hazard, cds_maturities, np.array([const_hazard]*len(cds_maturities)))

        # Sort input data by maturity
        sorted_indices = np.argsort(cds_maturities)
        cds_maturities_sorted = cds_maturities[sorted_indices]
        initial_cds_spreads_sorted = initial_cds_spreads[sorted_indices]

        # Calculate credit-adjusted discount factors and survival probabilities
        # This is a simplification; a more rigorous approach would be iterative.
        hazard_rates = []
        # Hazard rate for the first maturity
        if initial_cds_spreads_sorted[0] > 0 and (1 - self.recovery_rate) > 0:
            first_hazard = initial_cds_spreads_sorted[0] / (1 - self.recovery_rate)
            hazard_rates.append(first_hazard)
        else:
            hazard_rates.append(1e-6) # Small positive default

        # For subsequent maturities, approximate incremental hazard rate
        for i in range(1, len(cds_maturities_sorted)):
            T_curr = cds_maturities_sorted[i]
            T_prev = cds_maturities_sorted[i-1]
            S_curr = initial_cds_spreads_sorted[i]
            S_prev = initial_cds_spreads_sorted[i-1] # Previous spread

            if T_curr - T_prev > 1e-8 and (1 - self.recovery_rate) > 0:
                # Simple approximation: forward hazard rate from spread difference
                # This is highly simplified for a first pass
                h_curr = S_curr / (1 - self.recovery_rate)
                h_prev = S_prev / (1 - self.recovery_rate)
                # Use a linear interpolation in hazard space as a proxy for theta(t)
                # This is more an approximation of h(t) than rigorous theta(t)
                hazard_rates.append(h_curr + (h_curr - h_prev) * (T_curr - T_prev) / T_prev)
            else:
                hazard_rates.append(hazard_rates[-1]) # Use previous if invalid

        # Map these approximated hazard rates to t_values (the independent variable for theta)
        # Assuming theta(t) follows the shape of these implied hazard rates
        # Interpolate the derived hazard rates over the specified t_values range
        if len(cds_maturities_sorted) > 1 and len(hazard_rates) == len(cds_maturities_sorted):
            theta_lambda_interp_func = interp1d(cds_maturities_sorted, hazard_rates, kind='linear',
                                                fill_value=(hazard_rates[0], hazard_rates[-1]),
                                                bounds_error=False)
        else:
            # Fallback to constant theta if not enough points for interpolation
            constant_theta = np.mean(hazard_rates) if hazard_rates else 0.01
            theta_lambda_interp_func = lambda t: constant_theta

        print("  Finished deriving approximate theta_lambda(t).")
        return (theta_lambda_interp_func, cds_maturities_sorted, np.array(hazard_rates))


    # --- Credit Default Swap (CDS) Pricing Methods ---
    def generate_synthetic_cds_market_data(self, maturities: list[float] = None, base_spread_bps: float = 100,
                                           spread_curve_type: str = 'upward', noise_level: float = 5.0) -> pd.DataFrame:
        """
        Generates synthetic market CDS spread data for calibration purposes.

        Args:
            maturities (list[float], optional): List of CDS maturities in years (e.g., [1, 2, 3, 5, 7, 10]).
                                               Defaults to standard maturities if None.
            base_spread_bps (float): Base spread in basis points (e.g., 100 bps = 1%).
            spread_curve_type (str): 'flat', 'upward', or 'humped'. Determines curve shape.
            noise_level (float): Standard deviation of noise to add to spreads in basis points.

        Returns:
            pd.DataFrame: DataFrame with 'maturity' and 'spread' (in decimal).
        """
        print("\nGenerating synthetic CDS market data...")
        if maturities is None:
            maturities = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0] # Common CDS maturities

        spreads_bps = []
        for T in maturities:
            spread = base_spread_bps
            if spread_curve_type == 'upward':
                spread += T * 5 # Increase with maturity
            elif spread_curve_type == 'humped':
                spread += (T - 3)**2 * 5 - 10 # Hump around 3-year

            # Add random noise
            spread += np.random.normal(0, noise_level)
            spreads_bps.append(np.maximum(10, spread)) # Ensure spreads are at least 10 bps

        cds_data = pd.DataFrame({
            'maturity': maturities,
            'spread': np.array(spreads_bps) / 10000.0 # Convert bps to decimal
        })

        self.insert_simulated_cds_market_data(cds_data)
        self.simulated_cds_market_data = cds_data
        print(f"Generated {len(cds_data)} synthetic CDS spreads.")
        return cds_data

    def _simulate_paths_joint(self, T: float, num_paths: int, num_steps: int,
                              lambda0: float, kappa_lambda: float, sigma_lambda: float,
                              theta_lambda_interp_func: Callable,
                              r0: float, kappa_r: float, theta_r: float, sigma_r: float,
                              rho_correlation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates joint paths for short-term interest rate (r_t) and default intensity (lambda_t)
        using the Euler-Maruyama method for the CIR and Vasicek processes.

        Args:
            T (float): Total time to maturity.
            num_paths (int): Number of Monte Carlo paths.
            num_steps (int): Number of time steps.
            lambda0 (float): Initial default intensity.
            kappa_lambda (float): Speed of mean reversion for intensity.
            sigma_lambda (float): Volatility of intensity.
            theta_lambda_interp_func (Callable): Time-dependent mean intensity function for CIR++.
            r0 (float): Initial short interest rate.
            kappa_r (float): Speed of mean reversion for short rate (Vasicek).
            theta_r (float): Long-term mean for short rate (Vasicek).
            sigma_r (float): Volatility for short rate (Vasicek).
            rho_correlation (float): Correlation between interest rate and default intensity Wiener processes.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Array of simulated short rate paths (num_paths, num_steps + 1).
                - Array of simulated default intensity paths (num_paths, num_steps + 1).
                - Array of simulated survival times (num_paths,).
        """
        dt = T / num_steps

        # Initialize paths
        r_paths = np.zeros((num_paths, num_steps + 1))
        lambda_paths = np.zeros((num_paths, num_steps + 1))

        r_paths[:, 0] = r0
        lambda_paths[:, 0] = lambda0

        survival_times = np.full(num_paths, T) # Default to no default (maturity)

        # Generate correlated random numbers
        # dW1 and dW2 are standard normal, correlated with rho
        # dW_r = Z1
        # dW_lambda = rho * Z1 + sqrt(1 - rho^2) * Z2
        # where Z1, Z2 are independent N(0,1)
        z1_matrix = np.random.normal(0, 1, size=(num_paths, num_steps))
        z2_matrix = np.random.normal(0, 1, size=(num_paths, num_steps))

        dW_r_matrix = z1_matrix * np.sqrt(dt)
        dW_lambda_matrix = (rho_correlation * z1_matrix + np.sqrt(np.maximum(0, 1 - rho_correlation**2)) * z2_matrix) * np.sqrt(dt)

        for i in range(num_steps):
            t_current = i * dt
            # Get the time-dependent theta for CIR++
            theta_lambda_t = theta_lambda_interp_func(t_current)
            theta_lambda_t = np.maximum(theta_lambda_t, 1e-8) # Ensure positive drift target

            # Simulate next step for each path
            for j in range(num_paths):
                # Vasicek process for short rate (can be replaced with CIR for rates if desired)
                dr = kappa_r * (theta_r - r_paths[j, i]) * dt + sigma_r * dW_r_matrix[j, i]
                r_paths[j, i+1] = np.maximum(1e-8, r_paths[j, i] + dr) # Ensure rates are positive

                # CIR process for default intensity (Euler-Maruyama)
                # Ensure sqrt(lambda) is real and positive
                sqrt_lambda_ti = np.sqrt(np.maximum(lambda_paths[j, i], 1e-12)) # Small positive for sqrt

                d_lambda = kappa_lambda * (theta_lambda_t - lambda_paths[j, i]) * dt + sigma_lambda * sqrt_lambda_ti * dW_lambda_matrix[j, i]
                lambda_paths[j, i+1] = np.maximum(1e-12, lambda_paths[j, i] + d_lambda) # Ensure intensity remains positive


                if np.random.rand() < (lambda_paths[j, i+1] * dt):
                    if survival_times[j] == T: # If not already defaulted on this path
                        survival_times[j] = (i + 1) * dt # Record default time

        return r_paths, lambda_paths, survival_times

    def _calculate_model_spreads_numerical(self, kappa_l: float, sigma_l: float,
                                           maturities: np.ndarray, lambda0_fixed: float,
                                           theta_func_fixed: Callable, ir_params_fixed: dict,
                                           rec_rate_fixed: float,
                                           num_mc_paths: int = 500, num_mc_steps: int = 50) -> np.ndarray:
        """
        Helper function to calculate model-implied CDS spreads for given numerical parameters.
        This function is designed to be called by the PyMC custom likelihood, where
        kappa_l and sigma_l are numerical values.
        """
        model_spreads_arr = np.zeros(len(maturities))

        for i, maturity_t in enumerate(maturities):
            # Simulate joint paths for interest rates and default intensity using the *numerical* parameters
            r_paths, lambda_paths, survival_times = self._simulate_paths_joint(
                T=maturity_t, num_paths=num_mc_paths, num_steps=num_mc_steps,
                lambda0=lambda0_fixed, kappa_lambda=kappa_l, sigma_lambda=sigma_l,
                theta_lambda_interp_func=theta_func_fixed,
                r0=ir_params_fixed['r0'], kappa_r=ir_params_fixed['kappa_r'],
                theta_r=ir_params_fixed['theta_r'], sigma_r=ir_params_fixed['sigma_r'],
                rho_correlation=ir_params_fixed['rho_correlation']
            )

            # Calculate Protection Leg PV (per unit notional, unit (1-RR))
            pv_protection_calib = np.zeros(num_mc_paths)
            for k, default_time in enumerate(survival_times):
                if default_time < maturity_t:
                    # Adjusted index calculation to avoid out of bounds
                    time_step_length = maturity_t / num_mc_steps
                    idx_at_default = int(default_time / time_step_length)

                    # Take average around the calculated index
                    start_idx = max(0, idx_at_default - 1)
                    end_idx = min(num_mc_steps, idx_at_default + 1)

                    avg_r_at_default = np.mean(r_paths[k, start_idx : end_idx])

                    # Removed redundant np.isnan / np.isclose checks here.
                    # np.maximum calls in _simulate_paths_joint should ensure positivity.

                    df_to_default_time = self.get_discount_factor(default_time) # Use NSS-based DF here
                    if np.isnan(df_to_default_time): df_to_default_time = np.exp(-avg_r_at_default * default_time) # Fallback

                    pv_protection_calib[k] = (1 - rec_rate_fixed) * df_to_default_time
            expected_pv_protection_leg = np.mean(pv_protection_calib)

            # Calculate Premium Leg PV (per unit notional, per unit coupon)
            freq_payments = 4 # Quarterly for standard CDS
            # Ensure payment_times are correct based on maturity_t
            if maturity_t > 0:
                payment_times = np.linspace(1/freq_payments, maturity_t, int(maturity_t * freq_payments))
            else: # Handle zero maturity case, no payments
                payment_times = np.array([])

            pv_premium_leg_unit_coupon = np.zeros(num_mc_paths)
            for k in range(num_mc_paths):
                path_r = r_paths[k, :]
                default_time = survival_times[k]

                for t_idx, payment_time in enumerate(payment_times):
                    if payment_time <= default_time: # Only pay if not defaulted yet
                        # Use the average short rate over the interval for discounting
                        # Or a more robust discount factor from the current path
                        step_idx = int(payment_time / (maturity_t / num_mc_steps)) # Approximate step index for payment_time
                        step_idx = min(step_idx, num_mc_steps) # Ensure index is within bounds

                        # Use average short rate of the step for simplicity.
                        # More rigorously, integrate r_t over the interval.
                        avg_r_in_interval = np.mean(path_r[max(0, step_idx-1):step_idx+1])
                        # Removed redundant np.isnan / np.isclose checks here.

                        df_to_payment_time = self.get_discount_factor(payment_time) # Use NSS-based DF here
                        if np.isnan(df_to_payment_time): df_to_payment_time = np.exp(-avg_r_in_interval * payment_time) # Fallback

                        pv_premium_leg_unit_coupon[k] += (1.0 / freq_payments) * df_to_payment_time
                    else:
                        break # Defaulted before this payment, no further premium payments

                # Accrued premium calculation (simplified)
                # If default occurred and it's not exactly on a payment date
                if default_time < maturity_t and default_time not in payment_times:
                    # Find the last payment time before default_time
                    last_payment_time_before_default = 0.0
                    for pt_val in payment_times: # Renamed loop variable to avoid conflict with pytensor.tensor
                        if pt_val < default_time:
                            last_payment_time_before_default = pt_val
                        else:
                            break

                    # Pro-rata portion of the next coupon
                    time_since_last_payment = default_time - last_payment_time_before_default
                    if time_since_last_payment > 1e-9: # Ensure actual accrual period
                        accrual_factor = time_since_last_payment / (1.0 / freq_payments)
                        accrued_coupon_amount = (1.0 / freq_payments) * accrual_factor # As fraction of annual coupon

                        # Discount using rate at default time
                        # Use NSS DF for consistency if possible, else fallback to average rate from path
                        df_to_default_time = self.get_discount_factor(default_time)
                        if np.isnan(df_to_default_time): # Fallback using path average if NSS DF not available
                            time_step_length = maturity_t / num_mc_steps
                            idx_at_default = int(default_time / time_step_length)
                            start_idx = max(0, idx_at_default - 1)
                            end_idx = min(num_mc_steps, idx_at_default + 1)
                            avg_r_at_default = np.mean(path_r[start_idx : end_idx])
                            df_to_default_time = np.exp(-avg_r_at_default * default_time)

                        pv_premium_leg_unit_coupon[k] += accrued_coupon_amount * df_to_default_time


            expected_pv_premium_leg_unit_coupon = np.mean(pv_premium_leg_unit_coupon)

            # Calculate model-implied spread
            if expected_pv_premium_leg_unit_coupon > 1e-12: # Avoid division by zero/very small number
                model_spread = expected_pv_protection_leg / expected_pv_premium_leg_unit_coupon
            else:
                model_spread = np.array(1.0) # Assign a large arbitrary error if premium leg is near zero (will be heavily penalized)

            model_spreads_arr[i] = model_spread
        return model_spreads_arr

    def monte_carlo_cds_price(self, notional: float, maturity_T: float, coupon: float,
                              freq_payments_per_year: int, initial_cds_market_data: Optional[pd.DataFrame] = None,
                              num_paths: int = 10000, num_steps: int = 250) -> float:
        """
        Prices a Credit Default Swap (CDS) using Monte Carlo simulation.
        This function will internally calibrate the CIR++ model parameters for default intensity
        (kappa_lambda, sigma_lambda) to the provided initial CDS market data if they are not already set.
        The `theta_lambda_func` will also be derived from the initial market data.

        Args:
            notional (float): The notional amount of the CDS.
            maturity_T (float): The maturity of the CDS in years.
            coupon (float): The fixed annual premium payment (as a decimal, e.g., 0.01 for 100 bps).
            freq_payments_per_year (int): Frequency of premium payments per year (e.g., 4 for quarterly).
            initial_cds_market_data (pd.DataFrame, optional): DataFrame with 'maturity' and 'spread'
                                                               for calibrating the intensity model.
                                                               If None, tries to fetch from DB.
            num_paths (int): Number of Monte Carlo paths.
            num_steps (int): Number of time steps.

        Returns:
            float: The estimated Monte Carlo price of the CDS.
        """
        print(f"\nStarting Monte Carlo CDS pricing for Notional={notional:.2f}, Maturity={maturity_T:.2f}Y, Coupon={coupon:.4f}...")

        if initial_cds_market_data is None:
            initial_cds_market_data = self.get_simulated_cds_market_data()
            if initial_cds_market_data is None or initial_cds_market_data.empty:
                print("Error: No initial CDS market data available for intensity model calibration. Cannot price CDS.")
                return np.nan

        # Set initial lambda0 from the shortest maturity CDS spread
        # Simplified: lambda0 approx initial_spread / (1 - recovery_rate) for very short maturity
        if not initial_cds_market_data.empty:
            shortest_maturity_spread = initial_cds_market_data.iloc[0]['spread']
            self.lambda0 = shortest_maturity_spread / (1.0 - self.recovery_rate) if (1.0 - self.recovery_rate) > 1e-8 else shortest_maturity_spread
            self.lambda0 = np.maximum(self.lambda0, 1e-8)
            print(f"  Initial lambda0 set to: {self.lambda0:.6f}")
        else:
            self.lambda0 = 0.01 # Fallback if no market data

        # Derive theta_lambda_func for the CIR++ model from the market CDS spreads
        # This function will capture the initial term structure of credit.
        self.theta_lambda_func, _, _ = self._theta_lambda_calibration_target(
            t_values=np.linspace(0, maturity_T, num_steps + 1),
            initial_cds_spreads=initial_cds_market_data['spread'].values,
            cds_maturities=initial_cds_market_data['maturity'].values,
            r0=self.r0
        )

        # Calibrate CIR++ parameters (kappa_lambda, sigma_lambda) if not already set.
        # For simplicity in this comprehensive example, we'll use a hardcoded value if not calibrated.
        # In a real scenario, this would involve minimizing the error between model CDS prices and market prices.
        if self.kappa_lambda is None or self.sigma_lambda is None:
            print("  CIR++ intensity model parameters (kappa_lambda, sigma_lambda) are not calibrated.")
            print("  Using default values for demonstration purposes.")
            self.kappa_lambda = 0.5 # Default mean reversion speed
            self.sigma_lambda = 0.2 # Default volatility of intensity
            # In a real system, you'd call a dedicated calibration method here
            # self.calibrate_cds_model(initial_cds_market_data)
            # and then update self.kappa_lambda, self.sigma_lambda from calibrated values.

        print(f"  Simulating {num_paths} paths with {num_steps} steps for CDS pricing.")

        # Simulate joint paths for interest rates and default intensity
        r_paths, lambda_paths, survival_times = self._simulate_paths_joint(
            T=maturity_T, num_paths=num_paths, num_steps=num_steps,
            lambda0=self.lambda0, kappa_lambda=self.kappa_lambda, sigma_lambda=self.sigma_lambda,
            theta_lambda_interp_func=self.theta_lambda_func, # Corrected argument name here
            r0=self.r0, kappa_r=self.kappa_r, theta_r=self.theta_r, sigma_r=self.sigma_r,
            rho_correlation=self.rho_correlation
        )

        # Calculate Present Value of Premium Leg and Protection Leg for each path
        pv_premium_leg = np.zeros(num_paths)
        pv_protection_leg = np.zeros(num_paths)

        payment_times = np.linspace(0, maturity_T, int(maturity_T * freq_payments_per_year) + 1)[1:] # Exclude t=0

        for j in range(num_paths):
            path_r = r_paths[j, :]
            path_lambda = lambda_paths[j, :]
            default_time = survival_times[j]

            # --- Premium Leg ---
            # Protection buyer pays coupon at each payment date until maturity or default
            for t_idx, payment_time in enumerate(payment_times):
                if payment_time <= default_time: # Only pay if not defaulted yet
                    # Use the NSS discount factor for consistency with yield curve model
                    df_to_payment_time = self.get_discount_factor(payment_time)
                    if np.isnan(df_to_payment_time): # Fallback to path average if NSS DF unavailable
                        step_idx = int(payment_time / (maturity_T / num_steps))
                        step_idx = min(step_idx, num_steps)
                        avg_r_in_interval = np.mean(path_r[max(0, step_idx-1):step_idx+1])
                        df_to_payment_time = np.exp(-avg_r_in_interval * payment_time)

                    pv_premium_leg[j] += (coupon / freq_payments_per_year) * notional * df_to_payment_time

                else: # Defaulted before this payment, no further premium payments
                    # Add accrual payment for the period up to default
                    if default_time < payment_time and t_idx > 0: # If default occurred between last payment and current payment
                        last_payment_time = payment_times[t_idx-1]
                        accrued_period = default_time - last_payment_time
                        if accrued_period > 1e-9: # Only if actual accrual period
                            accrued_coupon_amount = (coupon / freq_payments_per_year) * (accrued_period / (1.0 / freq_payments_per_year)) * notional

                            # Use NSS discount factor
                            df_to_default_time = self.get_discount_factor(default_time)
                            if np.isnan(df_to_default_time): # Fallback using path average if NSS DF not available
                                step_idx_default = int(default_time / (maturity_T / num_steps))
                                step_idx_default = min(step_idx_default, num_steps)
                                avg_r_at_default = np.mean(path_r[max(0, step_idx_default-1):step_idx_default+1])
                                df_to_default_time = np.exp(-avg_r_at_default * default_time)

                            pv_premium_leg[j] += accrued_coupon_amount * df_to_default_time
                    break # Break from premium payment loop after default

            # --- Protection Leg ---
            # Protection seller pays (1 - Recovery Rate) * Notional at default
            if default_time < maturity_T: # If default occurred before maturity
                default_amount = notional * (1 - self.recovery_rate)
                # Discount using the NSS discount factor
                df_to_default_time = self.get_discount_factor(default_time)
                if np.isnan(df_to_default_time): # Fallback using path average if NSS DF not available
                    step_idx_default = int(default_time / (maturity_T / num_steps))
                    step_idx_default = min(step_idx_default, num_steps)
                    avg_r_at_default = np.mean(path_r[max(0, step_idx_default-1) : step_idx_default+1])
                    df_to_default_time = np.exp(-avg_r_at_default * default_time)

                pv_protection_leg[j] = default_amount * df_to_default_time

        # CDS Price = PV(Protection Leg) - PV(Premium Leg)
        estimated_cds_price = np.mean(pv_protection_leg) - np.mean(pv_premium_leg)

        # Store pricing result (as an example)
        cds_id = f"CDS_{notional}_{maturity_T}_{coupon}"
        self.insert_cds_pricing_result(
            cds_id, notional, maturity_T, coupon, self.recovery_rate, estimated_cds_price,
            'CIR++ MC',
            {'kappa_lambda': self.kappa_lambda, 'sigma_lambda': self.sigma_lambda, 'lambda0': self.lambda0},
            {'r0': self.r0, 'kappa_r': self.kappa_r, 'theta_r': self.theta_r, 'sigma_r': self.sigma_r}
        )

        print(f"  Monte Carlo CDS Price estimated: {estimated_cds_price:.4f}")
        return estimated_cds_price

    def calibrate_cds_model(self, initial_cds_market_data: pd.DataFrame,
                           initial_guess: Optional[list[float]] = None) -> Optional[dict[str, float]]:
        """
        Calibrates the stochastic default intensity model (CIR++) parameters
        (kappa_lambda, sigma_lambda) to market CDS spreads. This involves
        minimizing the sum of squared differences between model-implied CDS spreads
        and market CDS spreads.

        Args:
            initial_cds_market_data (pd.DataFrame): DataFrame with 'maturity' and 'spread' (decimal).
            initial_guess (list[float], optional): Initial guess for [kappa_lambda, sigma_lambda].
                                                  Defaults to [0.5, 0.2] if None.

        Returns:
            dict: Dictionary of calibrated parameters {kappa_lambda, sigma_lambda}, or None if failed.
        """
        print("\nStarting CIR++ CDS model calibration to market spreads (Least Squares)...")
        if initial_cds_market_data.empty:
            print("No CDS market data provided for calibration.")
            return None

        # Sort market data by maturity
        initial_cds_market_data = initial_cds_market_data.sort_values(by='maturity').reset_index(drop=True)

        # Set initial lambda0 from the shortest maturity CDS spread
        shortest_maturity_spread = initial_cds_market_data.iloc[0]['spread']
        self.lambda0 = shortest_maturity_spread / (1.0 - self.recovery_rate) if (1.0 - self.recovery_rate) > 1e-8 else shortest_maturity_spread
        self.lambda0 = np.maximum(self.lambda0, 1e-8) # Ensure positive

        # Derive theta_lambda_func (time-dependent mean)
        # We only need the callable here for the LS calibration part.
        self.theta_lambda_func, _, _ = self._theta_lambda_calibration_target(
            t_values=np.linspace(0, initial_cds_market_data['maturity'].max(), 100), # Grid for theta(t)
            initial_cds_spreads=initial_cds_market_data['spread'].values,
            cds_maturities=initial_cds_market_data['maturity'].values,
            r0=self.r0
        )

        if initial_guess is None:
            initial_guess = [0.5, 0.2] # [kappa_lambda, sigma_lambda]

        # Define the objective function for calibration
        def objective_function(params: list[float], market_data: pd.DataFrame) -> np.ndarray:
            kappa_l, sigma_l = params

            # Penalize invalid parameter values. If invalid, fill with large error and skip detailed calculation.
            if kappa_l <= 0 or sigma_l <= 0:
                # If parameters are invalid, return a large array of errors
                return np.full(len(market_data), 1e6)

            # Call the numerical calculation helper directly
            calculated_spreads = self._calculate_model_spreads_numerical(
                kappa_l=kappa_l, sigma_l=sigma_l,
                maturities=market_data['maturity'].values,
                lambda0_fixed=self.lambda0,
                theta_func_fixed=self.theta_lambda_func,
                ir_params_fixed={'r0': self.r0, 'kappa_r': self.kappa_r, 'theta_r': self.theta_r,
                                 'sigma_r': self.sigma_r, 'rho_correlation': self.rho_correlation},
                rec_rate_fixed=self.recovery_rate,
                num_mc_paths=10000, # Use more paths for LS calibration accuracy
                num_mc_steps=100 # Use more steps for LS calibration accuracy
            )

            # Return residuals (difference between market and model spreads)
            return market_data['spread'].values - calculated_spreads

        # End of objective_function (Least Squares)

        bounds = ([1e-6, 1e-6], [2.0, 1.0]) # Bounds for kappa_lambda (speed), sigma_lambda (vol)

        try:
            # Use least_squares for non-linear curve fitting
            result = least_squares(
                objective_function,
                initial_guess,
                bounds=bounds,
                args=(initial_cds_market_data,),
                method='trf', # Trust-Region-Reflective algorithm
                ftol=1e-6, xtol=1e-6, gtol=1e-6,
                max_nfev=500 # Max function evaluations
            )

            if result.success:
                calibrated_params = {'kappa_lambda': result.x[0], 'sigma_lambda': result.x[1]}
                self.kappa_lambda = calibrated_params['kappa_lambda']
                self.sigma_lambda = calibrated_params['sigma_lambda']

                # Store calibrated parameters in DB
                self.insert_model_parameters('CIR++_Intensity_LS', calibrated_params) # Changed name for LS
                print(f"CIR++ CDS model calibration (Least Squares) successful! Calibrated parameters: {calibrated_params}")
                return calibrated_params
            else:
                print(f"CIR++ CDS model calibration (Least Squares) failed: {result.message}")
                print(f"Last guess: {result.x}")
                return None
        except Exception as e:
            print(f"An error occurred during CIR++ CDS model calibration (Least Squares): {e}")
            return None

    def calibrate_cds_model_bayesian(self, initial_cds_market_data: pd.DataFrame,
                                     draws: int = 2000, tune: int = 1000, chains: int = 4) -> Optional[dict]:
        """
        Calibrates the stochastic default intensity model (CIR++) parameters
        (kappa_lambda, sigma_lambda) to market CDS spreads using Bayesian MCMC.

        Args:
            initial_cds_market_data (pd.DataFrame): DataFrame with 'maturity' and 'spread' (decimal).
            draws (int): Number of posterior draws per chain.
            tune (int): Number of tuning (warmup) steps per chain.
            chains (int): Number of independent MCMC chains.

        Returns:
            dict: Dictionary of posterior means and HPDI of calibrated parameters {kappa_lambda, sigma_lambda, sigma_obs},
                  or None if calibration fails.
        """
        print("\n--- Starting Bayesian CDS model calibration to market spreads using MCMC ---")
        if initial_cds_market_data.empty:
            print("No CDS market data provided for Bayesian calibration.")
            return None

        # Sort market data by maturity
        market_data_df_sorted = initial_cds_market_data.sort_values(by='maturity').reset_index(drop=True)
        observed_cds_spreads = market_data_df_sorted['spread'].values
        cds_maturities = market_data_df_sorted['maturity'].values

        # Set initial lambda0 from the shortest maturity CDS spread for theta_lambda_func derivation
        shortest_maturity_spread = market_data_df_sorted.iloc[0]['spread']
        self.lambda0 = shortest_maturity_spread / (1.0 - self.recovery_rate) if (1.0 - self.recovery_rate) > 1e-8 else shortest_maturity_spread
        self.lambda0 = np.maximum(self.lambda0, 1e-8)

        # Derive theta_lambda_func (time-dependent mean) and its underlying data
        self.theta_lambda_func, theta_maturities_data, theta_hazard_rates_data = self._theta_lambda_calibration_target(
            t_values=np.linspace(0, market_data_df_sorted['maturity'].max(), 100), # Grid for theta(t)
            initial_cds_spreads=observed_cds_spreads,
            cds_maturities=cds_maturities,
            r0=self.r0
        )

        # Instantiate the custom PyTensor Op
        cds_model_spread_op = CDSModelSpreadOp(self)

        # Define the custom log-probability function for the likelihood
        # This function will be called by PyMC during sampling, with symbolic PyTensor variables.
        def _logp_cds_model(observed_value: TensorVariable, # This argument is automatically filled by CustomDist's `observed` kwarg
                            kappa_l: TensorVariable, sigma_l: TensorVariable, sigma_obs: TensorVariable, # These are the symbolic variables being sampled
                            cds_maturities_array_fixed: np.ndarray, # Fixed data/parameters (NumPy arrays/scalars)
                            lambda0_fixed_val: float,
                            theta_maturities_fixed_val: np.ndarray,
                            theta_hazard_rates_fixed_val: np.ndarray,
                            r0_fixed_val: float, kappa_r_fixed_val: float, theta_r_fixed_val: float,
                            sigma_r_fixed_val: float, rho_correlation_fixed_val: float, # Flattened IR params
                            rec_rate_fixed_val: float) -> TensorVariable:

            # Use PyTensor's switch for conditional logic on symbolic variables

            cond_invalid_params = pt.or_(pt.le(kappa_l, 1e-8),
                                         pt.or_(pt.le(sigma_l, 1e-8), pt.le(sigma_obs, 1e-8)))

            # Call the custom PyTensor Op with symbolic variables.
            # This returns a symbolic PyTensor variable representing the model spreads.
            model_spreads_symbolic = cds_model_spread_op(
                kappa_l, sigma_l, lambda0_fixed_val,
                theta_maturities_fixed_val, theta_hazard_rates_fixed_val,
                r0_fixed_val, kappa_r_fixed_val, theta_r_fixed_val,
                sigma_r_fixed_val, rho_correlation_fixed_val,
                rec_rate_fixed_val, cds_maturities_array_fixed
            )

            # Compute log-likelihood using PyTensor's Normal distribution.
            # Both 'mu' (model_spreads_symbolic) and 'sigma' (sigma_obs) are symbolic.
            logp_val = pm.logp(
                pm.Normal.dist(mu=model_spreads_symbolic, sigma=sigma_obs),
                observed_value # This is the observed data from CustomDist
            )

            # Return negative infinity if parameters are invalid, otherwise the sum of log-likelihoods.
            # pt.sum is used to sum the log-likelihoods across all maturities.
            return pt.switch(cond_invalid_params, -np.inf, pt.sum(logp_val))
        # End of _logp_cds_model function

        # Define the PyMC model
        with pm.Model() as bayesian_cds_model:
            # Priors for kappa_lambda, sigma_lambda, and sigma_obs (observation noise)
            # HalfNormal ensures parameters are positive
            kappa_lambda = pm.HalfNormal("kappa_lambda", sigma=0.5) # Mean reversion speed for intensity
            sigma_lambda = pm.HalfNormal("sigma_lambda", sigma=0.2) # Volatility of intensity
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.001) # Observation noise for the market spreads

            # Custom likelihood for simulation-based models
            # Arguments for _logp_cds_model:
            # 1. `observed_value` (comes from `observed=observed_cds_spreads` keyword)
            # 2. `kappa_l` (comes from first positional arg `kappa_lambda`)
            # 3. `sigma_l` (comes from second positional arg `sigma_lambda`)
            # 4. `sigma_obs` (comes from third positional arg `sigma_obs`)
            # ... and so on for the fixed parameters passed as positional arguments.
            Y_obs = pm.CustomDist(
                "Y_obs",
                kappa_lambda,                 # Corresponds to `kappa_l` in _logp_cds_model
                sigma_lambda,                 # Corresponds to `sigma_l` in _logp_cds_model
                sigma_obs,                    # Corresponds to `sigma_obs` in _logp_cds_model
                # Fixed data/parameters passed as positional arguments, corresponding to arguments 4 onward in _logp_cds_model
                cds_maturities,               # Corresponds to `cds_maturities_array_fixed`
                self.lambda0,                 # Corresponds to `lambda0_fixed_val`
                theta_maturities_data,        # Corresponds to `theta_maturities_fixed_val`
                theta_hazard_rates_data,      # Corresponds to `theta_hazard_rates_fixed_val`
                # Pass individual IR parameters (these are float scalars)
                self.r0,                      # Corresponds to `r0_fixed_val`
                self.kappa_r,                 # Corresponds to `kappa_r_fixed_val`
                self.theta_r,                 # Corresponds to `theta_r_fixed_val`
                self.sigma_r,                 # Corresponds to `sigma_r_fixed_val`
                self.rho_correlation,         # Corresponds to `rho_correlation_fixed_val`
                self.recovery_rate,           # Corresponds to `rec_rate_fixed_val`
                logp=_logp_cds_model,         # The custom log-probability function
                observed=observed_cds_spreads, # The actual observed data for the likelihood
                shape=observed_cds_spreads.shape # Shape of the observed data
            )

        # Run MCMC sampling
        try:
            print(f"  Sampling {chains} chains, each with {tune} tune (warmup) steps and {draws} draws.")
            # `cores=1` is good practice for models with external (NumPy based) simulations
            # Using `nuts_sampler='blackjax'` or 'nutpie' can sometimes be more stable/faster
            trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, return_inferencedata=True, model=bayesian_cds_model)

            print("\nBayesian sampling completed. Analyzing results...")

            # Summarize posterior (mean, median, HPDIs)
            summary_stats = az.summary(trace, var_names=['kappa_lambda', 'sigma_lambda', 'sigma_obs'], kind="stats")
            print("\nPosterior Summary Statistics:")
            print(summary_stats)

            # Plot diagnostics (trace plots, posterior distributions)
            print("\nGenerating MCMC diagnostic plots...")
            az.plot_trace(trace, var_names=['kappa_lambda', 'sigma_lambda', 'sigma_obs'])
            plt.suptitle('Bayesian MCMC Trace Plots', fontsize=16)
            plt.tight_layout()
            plt.show()

            az.plot_posterior(trace, var_names=['kappa_lambda', 'sigma_lambda', 'sigma_obs'], kind='hist')
            plt.suptitle('Bayesian MCMC Posterior Distributions', fontsize=16)
            plt.tight_layout()
            plt.show()

            az.plot_pair(trace, var_names=['kappa_lambda', 'sigma_lambda', 'sigma_obs'], kind='kde', divergences=True)
            plt.suptitle('Bayesian MCMC Pair Plots (Parameter Correlations)', fontsize=16)
            plt.tight_layout()
            plt.show()

            # Update model's parameters with posterior means
            self.kappa_lambda = summary_stats.loc['kappa_lambda', 'mean']
            self.sigma_lambda = summary_stats.loc['sigma_lambda', 'mean']

            calibrated_params_bayesian = {
                'kappa_lambda_mean': self.kappa_lambda,
                'sigma_lambda_mean': self.sigma_lambda,
                'sigma_obs_mean': summary_stats.loc['sigma_obs', 'mean'],
                # Include HPDI for uncertainty quantification
                'kappa_lambda_hdi_3%': summary_stats.loc['kappa_lambda', 'hdi_3%'],
                'kappa_lambda_hdi_97%': summary_stats.loc['kappa_lambda', 'hdi_97%'],
                'sigma_lambda_hdi_3%': summary_stats.loc['sigma_lambda', 'hdi_3%'],
                'sigma_lambda_hdi_97%': summary_stats.loc['sigma_lambda', 'hdi_97%'],
                'sigma_obs_hdi_3%': summary_stats.loc['sigma_obs', 'hdi_3%'],
                'sigma_obs_hdi_97%': summary_stats.loc['sigma_obs', 'hdi_97%'],
            }
            # Store these parameters in your DB (only store the means for simplicity, or structure to store HPDI)
            # Filtering out HDI values to store only means in the database for simplicity
            params_to_store_db = {k:v for k,v in calibrated_params_bayesian.items() if not k.endswith(('%', 'hdi_3%', 'hdi_97%'))}
            self.insert_model_parameters('CIR++_Intensity_Bayesian', params_to_store_db)

            print(f"Bayesian CDS model calibration successful! Updated model parameters with posterior means:")
            print(f"  kappa_lambda: {self.kappa_lambda:.6f}")
            print(f"  sigma_lambda: {self.sigma_lambda:.6f}")
            print(f"  sigma_obs: {calibrated_params_bayesian['sigma_obs_mean']:.6f}")

            return calibrated_params_bayesian

        except Exception as e:
            print(f"An error occurred during Bayesian CDS model calibration: {e}")
            return None


    # --- Risky Bond Valuation ---
    def price_risky_bond(self, face_value: float, coupon_rate: float, maturity_T: float,
                         freq_payments_per_year: int, issue_date: datetime.date, valuation_date: datetime.date) -> float:
        """
        Prices a risky bond using credit-adjusted discounting, based on the fitted
        risk-free NSS yield curve and the calibrated default intensity model.
        Assumes fixed (calibrated) CIR++ parameters for simplicity.

        Args:
            face_value (float): The face value of the bond.
            coupon_rate (float): Annual coupon rate (as a decimal).
            maturity_T (float): Time to maturity in years from valuation date.
            freq_payments_per_year (int): Frequency of coupon payments per year.
            issue_date (datetime.date): The bond's issue date.
            valuation_date (datetime.date): The date at which to value the bond.

        Returns:
            float: The estimated risky bond price.
        """
        print(f"\nPricing risky bond (Face={face_value:.2f}, Coupon={coupon_rate:.4f}, Maturity={maturity_T:.2f}Y)...")
        if self.nss_params is None and not self.get_nss_params(): # Check NSS params directly
            print("Error: NSS yield curve parameters not available or not fitted.")
            return np.nan
        if self.kappa_lambda is None or self.sigma_lambda is None or self.lambda0 is None or self.theta_lambda_func is None:
            print("Error: Default intensity model parameters not calibrated for risky bond pricing.")
            return np.nan

        # Generate future payment dates
        payment_dates = []
        num_payments = int(maturity_T * freq_payments_per_year)
        for i in range(1, num_payments + 1):
            payment_time = i / freq_payments_per_year # Relative time from now
            payment_dates.append(payment_time)

        # Add final principal payment at maturity
        if maturity_T not in payment_dates:
             payment_dates.append(maturity_T)
        payment_dates = sorted(list(set(payment_dates))) # Remove duplicates and sort

        # Calculate expected present value of cash flows
        expected_pv = 0.0
        coupon_payment = face_value * coupon_rate / freq_payments_per_year

        # Iterate through each payment
        for t_payment in payment_dates:
            # Get risk-free discount factor from NSS curve
            df_risk_free = self.get_discount_factor(t_payment)

            # Explicitly handle cases where df_risk_free might be invalid (NaN or non-positive)
            if np.isnan(df_risk_free) or df_risk_free <= 0:
                print(f"  Warning: Could not get valid risk-free discount factor for T={t_payment:.2f} from NSS. Using 0 for this cash flow contribution.")
                df_risk_free = 0.0 # Set to 0 to ensure it's defined and doesn't contribute

            # Approximate survival probability using the expected average intensity up to t_payment
            # Use the derived theta_lambda_func to get a proxy for average hazard rate up to t_payment
            approx_hazard_rate = self.theta_lambda_func(t_payment) if self.theta_lambda_func else self.lambda0
            approx_hazard_rate = np.maximum(approx_hazard_rate, 1e-8) # Ensure positive

            survival_prob = np.exp(-approx_hazard_rate * t_payment)

            # Current Cash Flow
            cash_flow = coupon_payment
            if np.isclose(t_payment, maturity_T):
                cash_flow += face_value # Add principal at maturity

            # Accumulate expected present value
            expected_pv += cash_flow * survival_prob * df_risk_free



        print(f"  Estimated Risky Bond Price: {expected_pv:.4f}")
        return expected_pv

    # --- Visualization Methods ---
    def plot_yield_curve(self, curve_df: pd.DataFrame, title: str = "Fitted Yield Curve (NSS)"):
        """
        Plots the fitted zero-coupon yield curve and overlay market yields.

        Args:
            curve_df (pd.DataFrame): DataFrame with 'maturity', 'zero_rate', 'discount_factor' from NSS.
            title (str): Title of the plot.
        """
        if curve_df.empty:
            print("No yield curve data to plot.")
            return

        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        # Plot fitted zero rates
        plt.plot(curve_df['maturity'], curve_df['zero_rate'] * 100, linestyle='-', color='blue', label='Fitted NSS Zero Rate')

        # Overlay market par yields for comparison
        if self.market_bond_data is not None and not self.market_bond_data.empty:
            market_maturities = np.array([self._get_maturity_in_years(col.replace(' Yield', '')) for col in self.market_bond_data.iloc[0].filter(regex='Yield').dropna().index])
            market_yields = self.market_bond_data.iloc[0].filter(regex='Yield').dropna().values
            plt.plot(market_maturities, market_yields * 100, 'o', color='red', label='Market Par Yields')

        plt.title(f'{title} (Zero Rates & Market Par Yields)')
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.legend()


        plt.subplot(1, 2, 2)
        plt.plot(curve_df['maturity'], curve_df['discount_factor'], linestyle='-', color='red', label='Fitted NSS Discount Factor')
        plt.title(f'{title} (Discount Factors)')
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Discount Factor')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_simulated_intensity_paths(self, lambda_paths: np.ndarray, r_paths: np.ndarray,
                                       T: float, num_paths_to_plot: int = 5):
        """
        Plots a sample of simulated default intensity and short rate paths.

        Args:
            lambda_paths (np.ndarray): Array of simulated default intensity paths.
            r_paths (np.ndarray): Array of simulated short rate paths.
            T (float): Total time simulated.
            num_paths_to_plot (int): Number of random paths to plot.
        """
        if lambda_paths.shape[0] == 0:
            print("No intensity paths to plot.")
            return

        num_sim_paths = lambda_paths.shape[0]
        steps = lambda_paths.shape[1] - 1
        time_points = np.linspace(0, T, steps + 1)

        plt.figure(figsize=(14, 6))

        # Plot Default Intensity Paths
        plt.subplot(1, 2, 1)
        for i in np.random.choice(num_sim_paths, min(num_paths_to_plot, num_sim_paths), replace=False):
            plt.plot(time_points, lambda_paths[i, :], lw=1, alpha=0.7)
        plt.title('Simulated Default Intensity ($\lambda_t$) Paths')
        plt.xlabel('Time (Years)')
        plt.ylabel('Default Intensity')
        plt.grid(True)

        # Plot Short Rate Paths
        plt.subplot(1, 2, 2)
        # Check if r_paths actually contains non-zero data
        if not np.all(r_paths == 0):
            for i in np.random.choice(num_sim_paths, min(num_paths_to_plot, num_sim_paths), replace=False):
                plt.plot(time_points, r_paths[i, :] * 100, lw=1, alpha=0.7)
            plt.title('Simulated Short Rate ($r_t$) Paths')
            plt.xlabel('Time (Years)')
            plt.ylabel('Short Rate (%)')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Short Rate Paths Not Provided/Zero', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Simulated Short Rate ($r_t$) Paths (Not Applicable)')
            plt.xlabel('Time (Years)')
            plt.ylabel('Short Rate (%)')
            plt.grid(False) # Turn off grid if no data

        plt.tight_layout()
        plt.show()

    def plot_cds_spread_calibration(self, market_data_df: pd.DataFrame, model_params: dict, calib_type: str = 'Least Squares'):
        """
        Plots the market CDS spreads against the model-implied CDS spreads
        after calibration.

        Args:
            market_data_df (pd.DataFrame): DataFrame with 'maturity' and 'spread' from market.
            model_params (dict): Dictionary of calibrated CIR++ parameters {kappa_lambda, sigma_lambda}.
            calib_type (str): Type of calibration for plot title (e.g., 'Least Squares', 'Bayesian Mean').
        """
        if market_data_df.empty:
            print(f"No market CDS data to plot {calib_type} calibration results.")
            return
        if not model_params:
            print(f"No calibrated model parameters to plot for {calib_type}.")
            return

        print(f"\nPlotting CDS Spread Calibration Results ({calib_type})...")

        model_spreads_to_plot = [] # Use a new name to avoid conflicts
        for maturity in market_data_df['maturity']:
            # Use the appropriate parameter names from the dictionary
            k_lambda_val = model_params.get('kappa_lambda_mean', model_params.get('kappa_lambda'))
            s_lambda_val = model_params.get('sigma_lambda_mean', model_params.get('sigma_lambda'))

            if k_lambda_val is None or s_lambda_val is None:
                print(f"Warning: Missing kappa_lambda or sigma_lambda in model_params for plotting {calib_type}. Skipping.")
                continue

            # Reconstruct theta_lambda_func if necessary for plotting
            # This is important if self.theta_lambda_func was not set by the last calibration
            # or if it was set to a tuple for Bayesian calibration but expected to be a callable directly here.
            # In general, if self.theta_lambda_func is set to the callable and not the tuple, this works.
            # However, if it's currently a tuple (from Bayesian, which returns (func, mat, haz)), we need to handle.
            current_theta_func = self.theta_lambda_func
            if isinstance(current_theta_func, tuple):
                 # Assume the tuple structure (callable, maturities, hazard_rates)
                 current_theta_func = current_theta_func[0]
            elif current_theta_func is None:
                # Fallback if theta_lambda_func is not set at all
                # This should ideally be handled during a calibration run before plotting
                print(f"Warning: theta_lambda_func is not set. Cannot plot calibration for {calib_type}.")
                continue

            # Call the numerical calculation helper directly
            # For plotting, we can use slightly higher MC paths for smoothness if needed, or stick to calibration settings.
            model_implied_spread_single = self._calculate_model_spreads_numerical(
                kappa_l=k_lambda_val, sigma_l=s_lambda_val,
                maturities=np.array([maturity]), # Pass a single maturity as an array
                lambda0_fixed=self.lambda0,
                theta_func_fixed=current_theta_func, # Use the actual callable
                ir_params_fixed={'r0': self.r0, 'kappa_r': self.kappa_r, 'theta_r': self.theta_r,
                                 'sigma_r': self.sigma_r, 'rho_correlation': self.rho_correlation},
                rec_rate_fixed=self.recovery_rate,
                num_mc_paths=5000, # More paths for smoother plotting
                num_mc_steps=100
            )[0] # Get the single spread value back

            model_spreads_to_plot.append(model_implied_spread_single)

        print(f"DEBUG: Length of market_data_df['maturity']: {len(market_data_df['maturity'])}")
        print(f"DEBUG: Length of model_spreads_to_plot: {len(model_spreads_to_plot)}")

        plt.figure(figsize=(10, 6))
        plt.plot(market_data_df['maturity'], market_data_df['spread'] * 10000, 'o', label='Market Spreads (bps)')
        plt.plot(market_data_df['maturity'], np.array(model_spreads_to_plot) * 10000, '-', label=f'{calib_type} Model Implied Spreads (bps)', color='red')
        plt.title(f'CDS Spread Calibration: Market vs. CIR++ Model ({calib_type})')
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Spread (Basis Points)')
        plt.legend()
        plt.grid(True)
        plt.show()


# --- Main Execution Block (for demonstration and testing) ---
if __name__ == "__main__":
    print("Starting Credit Risk Modeling and CDS Pricing Project...")

    # Set your FRED API key here, or ensure it's set as an environment variable (recommended)
    # You can get a free FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html
    # Example: fred_api_key = "YOUR_FRED_API_KEY"
    fred_api_key = "1b1d907f5ca95e001c785aada807996f" # <--- IMPORTANT: REPLACE None WITH YOUR ACTUAL FRED API KEY STRING HERE ---
    # For example: fred_api_key = "abcdef1234567890abcdef12345678"

    # Initialize the model
    model = CreditRiskModel(fred_api_key=fred_api_key, db_name='credit_risk_model.db')

    # --- Part 1: Yield Curve Modeling (Nelson-Siegel-Svensson) & SQLite Integration ---
    print("\n--- Part 1: Yield Curve Modeling (Nelson-Siegel-Svensson) & SQLite Integration ---")

    # 1. Fetch Treasury Yields
    today = datetime.date.today()
    # Increased fetch window from 30 to 90 days for more robustness
    start_date_fred = (today - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
    end_date_fred = today.strftime('%Y-%m-%d')

    latest_treasury_yields = model.fetch_treasury_yields(start_date=start_date_fred, end_date=end_date_fred)

    if latest_treasury_yields is None or latest_treasury_yields.empty:
        print("\nFATAL ERROR: Could not fetch Treasury yields. Exiting. Please check FRED API key and internet connection.")
        exit()

    # 2. Fit the Nelson-Siegel-Svensson Yield Curve
    fitted_nss_curve_df = model.fit_nss_yield_curve(latest_treasury_yields)

    if fitted_nss_curve_df is None or fitted_nss_curve_df.empty:
        print("\nFATAL ERROR: NSS yield curve fitting failed. Exiting.")
        exit()

    # 3. Retrieve and Verify Data from DB (NSS params)
    retrieved_nss_params_from_db = model.get_nss_params()
    if retrieved_nss_params_from_db is not None:
        print("\nSuccessfully retrieved NSS parameters from DB:")
        print(retrieved_nss_params_from_db)
        # Plot the fitted NSS curve, which now also overlays market data
        model.plot_yield_curve(fitted_nss_curve_df, title="Fitted Nelson-Siegel-Svensson Yield Curve")

    # Example: Get a specific discount factor using the NSS curve
    df_1y = model.get_discount_factor(1.0)
    df_5y = model.get_discount_factor(5.0)
    print(f"\nDiscount Factor for 1 year (NSS): {df_1y:.4f}")
    print(f"Discount Factor for 5 years (NSS): {df_5y:.4f}")
    print(f"Zero Rate for 1 year (NSS): {model.get_zero_rate(1.0)*100:.4f}%")
    print(f"Zero Rate for 5 years (NSS): {model.get_zero_rate(5.0)*100:.4f}%")


    # --- Part 2: Simulating CDS Market Data ---
    print("\n--- Part 2: Simulating CDS Market Data ---")
    # Generate some synthetic CDS spreads for different maturities
    sim_cds_data = model.generate_synthetic_cds_market_data(
        maturities=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        base_spread_bps=120,
        spread_curve_type='upward',
        noise_level=7.5 # Add some noise for realism
    )
    print("\nSimulated CDS Market Data:")
    print(sim_cds_data)

    # Retrieve and verify from DB
    retrieved_sim_cds_data = model.get_simulated_cds_market_data()
    if retrieved_sim_cds_data is not None:
        print("\nSuccessfully retrieved simulated CDS market data from DB:")
        print(retrieved_sim_cds_data)


    # --- Part 3: Calibrate Stochastic Default Intensity Model (CIR++) ---
    print("\n--- Part 3: Calibrating CIR++ Default Intensity Model ---")

    # Perform Least Squares Calibration (as before)
    print("\n--- Running Least Squares Calibration ---")
    calibrated_cir_params_ls = model.calibrate_cds_model(initial_cds_market_data=sim_cds_data)

    if calibrated_cir_params_ls:
        # Note: model's kappa_lambda and sigma_lambda are now set by LS calibration.
        # This will be overwritten if Bayesian calibration is run next.
        retrieved_cir_params_ls_db = model.get_model_parameters('CIR++_Intensity_LS')
        print(f"\nRetrieved calibrated CIR++ (Least Squares) parameters from DB: {retrieved_cir_params_ls_db}")
        model.plot_cds_spread_calibration(sim_cds_data, retrieved_cir_params_ls_db, calib_type='Least Squares')
    else:
        print("\nSkipping Least Squares CDS model calibration plot as calibration failed.")

    # Perform Bayesian Calibration (NEW)
    print("\n--- Running Bayesian Calibration ---")
    # Set more draws and tune steps for better MCMC convergence, but keep `cores=1` for safety.
    calibrated_cir_params_bayesian = model.calibrate_cds_model_bayesian(
        initial_cds_market_data=sim_cds_data,
        draws=5000, # Number of posterior samples  <-- ADJUST THIS TO 2000 FOR FULL RUN
        tune=2000,  # Number of warmup samples (discarded) <-- ADJUST THIS TO 1000 FOR FULL RUN
        chains=2    # Number of independent chains
    )

    if calibrated_cir_params_bayesian:
        # Model's kappa_lambda and sigma_lambda are now set by Bayesian calibration (posterior means).
        # You can retrieve them from DB as well:
        retrieved_cir_params_bayesian_db = model.get_model_parameters('CIR++_Intensity_Bayesian')
        print(f"\nRetrieved calibrated CIR++ (Bayesian Mean) parameters from DB: {retrieved_cir_params_bayesian_db}")
        model.plot_cds_spread_calibration(sim_cds_data, retrieved_cir_params_bayesian_db, calib_type='Bayesian Mean')
    else:
        print("\nSkipping Bayesian CDS model calibration plot as calibration failed.")


    # --- Part 4: Monte Carlo CDS Pricing ---
    print("\n--- Part 4: Monte Carlo CDS Pricing ---")
    # Example CDS to price
    cds_notional = 1_000_000.0
    cds_maturity = 5.0 # 5-year CDS
    cds_coupon = 0.0150 # 150 bps annual coupon
    cds_freq = 4 # Quarterly payments

    # Ensure model parameters are set from calibration or defaults
    # (The current `model.kappa_lambda` and `model.sigma_lambda` will be from the *last successful calibration*,
    # which is the Bayesian one if it ran. If not, they remain from LS or initial defaults.)
    if model.kappa_lambda is None: model.kappa_lambda = 0.5
    if model.sigma_lambda is None: model.sigma_lambda = 0.2
    if model.lambda0 is None:
        # Fallback for lambda0 if calibration was skipped entirely
        model.lambda0 = sim_cds_data.iloc[0]['spread'] / (1.0 - model.recovery_rate) if not sim_cds_data.empty and (1.0 - model.recovery_rate) > 1e-8 else 0.01

    if model.theta_lambda_func is None or isinstance(model.theta_lambda_func, tuple):
        # Re-derive theta_lambda_func to ensure it's a callable for Monte Carlo pricing,
        # in case it was left as a tuple from the Bayesian calibration
        model.theta_lambda_func, _, _ = model._theta_lambda_calibration_target(
            t_values=np.linspace(0, cds_maturity, 100),
            initial_cds_spreads=sim_cds_data['spread'].values,
            cds_maturities=sim_cds_data['maturity'].values,
            r0=model.r0
        )


    if model.kappa_lambda is not None and model.sigma_lambda is not None:
        mc_cds_price = model.monte_carlo_cds_price(
            notional=cds_notional,
            maturity_T=cds_maturity,
            coupon=cds_coupon,
            freq_payments_per_year=cds_freq,
            initial_cds_market_data=sim_cds_data, # Pass to ensure theta_lambda_func is consistent
            num_paths=50000, # More paths for final pricing
            num_steps=500 # More steps for accuracy
        )
        print(f"\nFinal Monte Carlo CDS Price (Notional={cds_notional:.0f}, Maturity={cds_maturity}Y, Coupon={cds_coupon*10000:.0f}bps): {mc_cds_price:.2f}")

        # Retrieve and verify CDS pricing result from DB
        retrieved_pricing_results = model.get_cds_pricing_results()
        if retrieved_pricing_results is not None:
            print("\nSuccessfully retrieved CDS pricing results from DB:")
            print(retrieved_pricing_results.tail())

        # Plot simulated paths (using the parameters from the last MC run, which should be Bayesian means)
        # Re-run a small simulation to get paths for plotting without affecting pricing
        r_paths_plot, lambda_paths_plot, _ = model._simulate_paths_joint( # Also get r_paths for plot
            T=cds_maturity, num_paths=10, num_steps=500, # Few paths for plotting
            lambda0=model.lambda0, kappa_lambda=model.kappa_lambda, sigma_lambda=model.sigma_lambda,
            theta_lambda_interp_func=model.theta_lambda_func, # Corrected argument name here
            r0=model.r0, kappa_r=model.kappa_r, theta_r=model.theta_r, sigma_r=model.sigma_r,
            rho_correlation=model.rho_correlation
        )
        model.plot_simulated_intensity_paths(
            lambda_paths_plot,
            r_paths=r_paths_plot, # Pass actual r_paths
            T=cds_maturity, num_paths_to_plot=5
        )

    else:
        print("\nSkipping Monte Carlo CDS pricing as model parameters are not set.")


    # --- Part 5: Risky Bond Valuation ---
    print("\n--- Part 5: Risky Bond Valuation ---")
    bond_face_value = 1000.0
    bond_coupon_rate = 0.03 # 3% annual coupon
    bond_maturity_years = 7.0
    bond_freq = 2 # Semi-annual payments

    # Dates for bond valuation (assuming today as valuation date)
    valuation_date = datetime.date.today()
    issue_date = valuation_date - datetime.timedelta(days=365*2) # Issued 2 years ago

    if model.kappa_lambda is not None and model.sigma_lambda is not None:
        risky_bond_price = model.price_risky_bond(
            face_value=bond_face_value,
            coupon_rate=bond_coupon_rate,
            maturity_T=bond_maturity_years,
            freq_payments_per_year=bond_freq,
            issue_date=issue_date,
            valuation_date=valuation_date
        )
        print(f"Risky Bond Price (Face={bond_face_value:.2f}, Coupon={bond_coupon_rate*100:.2f}%, Maturity={bond_maturity_years}Y): {risky_bond_price:.4f}")
    else:
        print("Skipping risky bond valuation as default intensity model not calibrated.")

    print("\nCredit Risk Modeling Project completed!")

