"""
Data splitting utilities for creating smaller test datasets
"""
import os
import pandas as pd
import random
from typing import List, Tuple, Optional
import argparse


class DataSplitter:
    """Utility class for splitting large datasets into smaller test datasets"""
    
    def __init__(self, base_dir: str = "./data", output_dir: str = "./data/test_small"):
        """
        Initialize data splitter
        
        Args:
            base_dir: Base directory containing original datasets
            output_dir: Output directory for small test datasets
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.sato_cv_dir = os.path.join(base_dir, "sato_cv")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "sato_cv"), exist_ok=True)
        
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(output_dir)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
    
    def split_sato_dataset(self, 
                          num_tables: int = 50, 
                          num_columns_per_table: int = 5,
                          cv_folds: List[int] = [0, 1, 2, 3, 4],
                          random_seed: int = 42) -> None:
        """
        Split SATO dataset into smaller test datasets
        
        Args:
            num_tables: Number of tables to include in test dataset
            num_columns_per_table: Maximum number of columns per table
            cv_folds: Which CV folds to process
            random_seed: Random seed for reproducibility
        """
        random.seed(random_seed)
        
        print(f"Creating small test dataset with {num_tables} tables...")
        
        for cv in cv_folds:
            input_file = os.path.join(self.sato_cv_dir, f"sato_cv_{cv}.csv")
            output_file = os.path.join(self.output_dir, "sato_cv", f"sato_cv_{cv}.csv")
            
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping...")
                continue
            
            print(f"Processing CV fold {cv}...")
            self._split_single_cv_file(input_file, output_file, num_tables, num_columns_per_table)
        
        print(f"Small test dataset created in {self.output_dir}")
    
    def _split_single_cv_file(self, 
                             input_file: str, 
                             output_file: str, 
                             num_tables: int, 
                             max_columns: int) -> None:
        """
        Split a single CV file
        
        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path
            num_tables: Number of tables to include
            max_columns: Maximum columns per table
        """
        # Read the full dataset
        df = pd.read_csv(input_file)
        
        # Get unique table IDs
        unique_tables = df['table_id'].unique()
        
        # Randomly sample tables
        if len(unique_tables) > num_tables:
            selected_tables = random.sample(list(unique_tables), num_tables)
        else:
            selected_tables = list(unique_tables)
        
        # Filter data for selected tables
        filtered_df = df[df['table_id'].isin(selected_tables)]
        
        # Limit columns per table
        result_rows = []
        for table_id in selected_tables:
            table_data = filtered_df[filtered_df['table_id'] == table_id]
            
            # Limit number of columns per table
            if len(table_data) > max_columns:
                # Sample columns randomly
                table_data = table_data.sample(n=max_columns, random_state=42)
            
            result_rows.append(table_data)
        
        # Combine and save
        result_df = pd.concat(result_rows, ignore_index=True)
        result_df.to_csv(output_file, index=False)
        
        print(f"  Created {output_file} with {len(result_df)} rows from {len(selected_tables)} tables")
    
    def create_test_config(self, 
                          config_name: str = "test_small.yaml",
                          num_tables: int = 50,
                          batch_size: int = 2,
                          num_epochs: int = 1) -> str:
        """
        Create a test configuration file for small dataset
        
        Args:
            config_name: Name of the config file
            num_tables: Number of tables in test dataset
            batch_size: Batch size for testing
            num_epochs: Number of epochs for testing
            
        Returns:
            Path to created config file
        """
        config_path = os.path.join(self.output_dir, config_name)
        
        config_content = f"""# Test configuration for small dataset
naive_pll:
  # Training parameters (reduced for testing)
  num_train_epochs: {num_epochs}
  batch_size: {batch_size}
  learning_rate: 5e-5
  warmup_ratio: 0.0
  
  # PLL specific parameters
  loss_function: "uniform_pll"
  num_candidates: 3  # Reduced for testing
  noise_ratio: 0.3
  
  # Model parameters
  max_length: 128
  num_classes: 78
  
  # Data parameters (pointing to small test dataset)
  dataset_name: "sato0"
  data_dir: "{self.output_dir}"
  multicol_only: false
  
  # Evaluation parameters
  eval_batch_size: 4  # Reduced for testing
  save_predictions: true
  output_dir: "./output/naive_pll_test"
  
  # Random seed
  random_seed: 42
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"Test configuration created: {config_path}")
        return config_path
    
    def get_dataset_stats(self, data_dir: Optional[str] = None) -> dict:
        """
        Get statistics about the dataset
        
        Args:
            data_dir: Directory to analyze (default: self.output_dir)
            
        Returns:
            Dictionary with dataset statistics
        """
        if data_dir is None:
            data_dir = self.output_dir
        
        stats = {}
        sato_cv_dir = os.path.join(data_dir, "sato_cv")
        
        if not os.path.exists(sato_cv_dir):
            return {"error": "sato_cv directory not found"}
        
        for cv in range(5):
            file_path = os.path.join(sato_cv_dir, f"sato_cv_{cv}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                stats[f"cv_{cv}"] = {
                    "total_rows": len(df),
                    "unique_tables": df['table_id'].nunique(),
                    "unique_classes": df['class'].nunique(),
                    "avg_columns_per_table": len(df) / df['table_id'].nunique() if df['table_id'].nunique() > 0 else 0
                }
        
        return stats


def main():
    """Command line interface for data splitting"""
    parser = argparse.ArgumentParser(description="Split large datasets into smaller test datasets")
    parser.add_argument("--num_tables", type=int, default=50, help="Number of tables in test dataset")
    parser.add_argument("--num_columns", type=int, default=15, help="Max columns per table")
    parser.add_argument("--cv_folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="CV folds to process")
    parser.add_argument("--base_dir", type=str, default="./data", help="Base data directory")
    parser.add_argument("--output_dir", type=str, default="./data/test_small", help="Output directory")
    parser.add_argument("--create_config", action="store_true", help="Create test configuration file")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create data splitter
    splitter = DataSplitter(args.base_dir, args.output_dir)
    
    if args.stats:
        # Show statistics
        print("Original dataset statistics:")
        original_stats = splitter.get_dataset_stats(args.base_dir)
        for cv, stats in original_stats.items():
            if isinstance(stats, dict):
                print(f"  {cv}: {stats['total_rows']} rows, {stats['unique_tables']} tables")
        
        print("\nTest dataset statistics:")
        test_stats = splitter.get_dataset_stats(args.output_dir)
        for cv, stats in test_stats.items():
            if isinstance(stats, dict):
                print(f"  {cv}: {stats['total_rows']} rows, {stats['unique_tables']} tables")
    else:
        # Split dataset
        splitter.split_sato_dataset(
            num_tables=args.num_tables,
            num_columns_per_table=args.num_columns,
            cv_folds=args.cv_folds,
            random_seed=args.random_seed
        )
        
        if args.create_config:
            splitter.create_test_config()
        
        # Show final statistics
        print("\nFinal test dataset statistics:")
        stats = splitter.get_dataset_stats()
        for cv, stat in stats.items():
            if isinstance(stat, dict):
                print(f"  {cv}: {stat['total_rows']} rows, {stat['unique_tables']} tables, "
                      f"{stat['avg_columns_per_table']:.1f} avg columns/table")


if __name__ == "__main__":
    main()
