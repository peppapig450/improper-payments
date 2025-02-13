from pathlib import Path
from typing import Final, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def clean_fraud_amount(amount: str) -> float:
    """
    Clean and convert fraud amount strings to float values.
    Handles dollar signs, commas, and converts dash to zero.

    Args:
        amount: Raw string amount from CSV
    Returns:
        Cleaned amount in millions of dollars
    """
    return float(amount.replace("$", "").replace(",", "").replace("-", "0")) * 1_000_000


def load_and_clean_data(file_path: Path | str) -> pd.DataFrame:
    """
    Load fraud data from CSV and clean the Confirmed Fraud column.

    Args:
        file_path: Path to the CSV file
    Returns:
        Cleaned DataFrame with properly formatted fraud amounts
    """
    try:
        df = pd.read_csv(file_path)

        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        df["Confirmed Fraud"] = df["Confirmed Fraud"].apply(clean_fraud_amount)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find confirmed fraud data at {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty")


def aggregate_fraud_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate fraud amounts by agency and program.

    Args:
        df: Clean DataFrame with fraud data
    Returns:
        Tuple of (agency_aggregation, program_aggregation) DataFrames
    """
    fraud_by_agency = (
        df.groupby("Agency", observed=True)["Confirmed Fraud"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fraud_by_program = (
        df.groupby(["Agency", "Program or Activity"], observed=True)["Confirmed Fraud"]
        .sum()
        .reset_index()
    )

    return fraud_by_agency, fraud_by_program


class FraudVisualization:
    """
    Class to handle interactive fraud visualizations with click events.
    """

    # Constants
    MILLION: Final[int] = 1_000_000
    FIGURE_SIZE_LARGE: Final[tuple[int, int]] = (12, 8)
    FIGURE_SIZE_MEDIUM: Final[tuple[int, int]] = (12, 6)

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fraud_by_agency, self.fraud_by_program = aggregate_fraud_data(df)
        self.current_figure = None
        self.program_figure = None

    def format_amount(self, x: float, p: Any) -> str:
        """Format large numbers into millions/billions with proper notation."""
        if x >= 1e9:
            return f"${x / 1e9:.1f}B"
        return f"${x / self.MILLION:.0f}M"

    def plot_top_agencies(self, n_agencies: int = 10) -> None:
        """Create initial bar plot of top agencies."""
        self.current_figure = plt.figure(figsize=(12, 6))

        ax = sns.barplot(
            data=self.fraud_by_agency.head(n_agencies),
            x="Confirmed Fraud",
            y="Agency",
            palette="Reds_r",
        )

        ax.set(
            xlabel="Total Fraud Amount ($)",
            ylabel="Agency",
            title=f"Top {n_agencies} Agencies with Highest Confirmed Fraud\nClick on a bar to see program details",
        )
        plt.xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_amount))

        # Connect the click event
        self.current_figure.canvas.mpl_connect("button_press_event", self.on_bar_click)
        plt.tight_layout()

    def plot_agency_programs(self, agency: str, n_programs: int = 10) -> None:
        """
        Create bar plot of top programs for a specific agency.

        Args:
            agency: Name of the agency to show programs for
            n_programs: Number of top programs to display
        """
        # Get programs for the selected agency
        agency_programs = (
            self.fraud_by_program[self.fraud_by_program["Agency"] == agency]
            .sort_values("Confirmed Fraud", ascending=False)
            .head(n_programs)
        )

        # Create new figures for programs
        if self.program_figure is not None:
            plt.close(self.program_figure)

        self.program_figure = plt.figure(figsize=self.FIGURE_SIZE_MEDIUM)
        ax = sns.barplot(
            data=agency_programs,
            x="Confirmed Fraud",
            y="Program or Activity",
            palette="Blues_r",
        )

        ax.set(
            xlabel="Total Fraud Amount ($)",
            ylabel="Program or Activity",
            title=f"Top {n_programs} Programs with Highest Confirmed Fraud\nAgency: {agency}",
        )

        plt.xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_amount))
        plt.tight_layout()
        plt.show()

    def on_bar_click(self, event) -> None:
        """Handle click events on the agency bar chart."""
        if event.inaxes is None:
            return

        ax = event.inaxes
        for i, bar in enumerate(ax.patches):
            if bar.contains(event)[0]:
                # Get the agency name from the y-axis labels
                agency = ax.get_yticklabels()[i].get_text()
                self.plot_agency_programs(agency)
                break

def main(file_path: str | Path = "data/gov_fraud_data.csv") -> None:
    """
    Main function to run the fraud analysis and create visualizations.

    Args:
        file_path: Path to the fraud data CSV file
    """
    try:
        # Load and process data
        df = load_and_clean_data(file_path)

        # Create visualizations
        viz = FraudVisualization(df)
        viz.plot_top_agencies()

        # plot_fraud_heatmap(df)

        plt.show()

    except Exception as e:
        print(f"Error processing fraud data: {e}")


if __name__ == "__main__":
    main()
