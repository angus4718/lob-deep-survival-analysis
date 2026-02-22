"""
Dataset post-processing for survival analysis.

This module loads raw tracked orders from Parquet, applies labeling,
and returns a fully labeled dataset ready for modeling.
"""

from typing import Optional, Dict, Any
import pandas as pd

from .competing_risks import ExecutionCompetingRisksLabeler
from .base import BaseLabeler


class SurvivalDataset:
    """
    Post-processes raw order tracking output with event labeling.
    
    Pipeline:
    1. Load raw orders from Parquet (output of OrderTracker.process_stream)
    2. Apply BaseLabeler subclass to each order
    3. Return enriched DataFrame with event_type, event_time_bin, etc.
    """
    
    def __init__(self, labeler: Optional[BaseLabeler] = None):
        """
        Args:
            labeler: BaseLabeler instance (e.g., ExecutionCompetingRisksLabeler).
                     Defaults to ExecutionCompetingRisksLabeler with CONFIG defaults.
        """
        self.labeler = labeler or ExecutionCompetingRisksLabeler()
    
    def load_and_label(
        self, 
        parquet_path: str,
        output_parquet_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load raw orders and apply labeling.
        
        Args:
            parquet_path: Path to raw Parquet file from OrderTracker.
            output_parquet_path: Optional path to save labeled DataFrame.
        
        Returns:
            DataFrame with original columns + ['event_type', 'event_time_bin'].
        """
        df = pd.read_parquet(parquet_path)
        
        labels = []
        for _, row in df.iterrows():
            insertion_context = row.to_dict()
            
            result = self.labeler.label(insertion_context)
            
            labels.append({
                "event_type": result["event_type"],
                "event_time_bin": result["event_time_bin"],
                **result.get("extras", {}),
            })
        
        labels_df = pd.DataFrame(labels)
        df = pd.concat([df.reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1)
        
        # Save if requested
        if output_parquet_path:
            df.to_parquet(output_parquet_path, index=False)
            print(f"Labeled dataset saved to {output_parquet_path}")
        
        return df
    
    def get_event_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Return counts of each event type."""
        return df["event_type"].value_counts().to_dict()
    
    def get_time_bin_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return statistics on time bins."""
        return {
            "min_bin": df["event_time_bin"].min(),
            "max_bin": df["event_time_bin"].max(),
            "median_bin": df["event_time_bin"].median(),
        }
