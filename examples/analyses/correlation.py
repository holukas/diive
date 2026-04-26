"""
Examples for daily correlation analysis.

Run this script to see example plots:
    python examples/analyses/correlation.py
"""
import diive as dv
from diive.pkgs.createvar.potentialradiation import potrad


def example_daily_correlation_radiation():
    """Daily correlation between measured and potential radiation."""
    df = dv.load_exampledata_parquet()

    # Use July 2022 data
    keeplocs = (df.index.year == 2022) & (df.index.month == 7)
    df = df.loc[keeplocs].copy()

    series = df['Rg_f'].copy()

    # Calculate potential radiation
    reference = potrad(
        timestamp_index=series.index,  # type: ignore
        lat=47.286417,
        lon=7.733750,
        utc_offset=1
    )

    # Calculate daily correlation
    corr = dv.DailyCorrelation(s1=series, s2=reference, mincorr=0.8)

    # All daily correlations
    all_corrs = corr.correlations
    print(f"\nDaily correlations (n={len(all_corrs)} days):")
    print(all_corrs.head())

    # Summary statistics
    print("\nSummary statistics:")
    summary = corr.summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
        else:
            print(f"  {key:.<30} {value}")

    # Best and worst correlation days
    print("\nHighest correlations (best match):")
    best = corr.get_days_by_correlation(high=True)
    print(best.head(5))

    print("\nLowest correlations (worst match):")
    worst = corr.get_days_by_correlation(high=False)
    print(worst.head(5))

    # Detect anomalous days
    anomalies = corr.detect_anomalies(method='zscore', threshold=2.0)
    print(f"\nAnomalies detected: {anomalies['is_anomaly'].sum()}/{len(anomalies)}")
    if anomalies['is_anomaly'].any():
        print(anomalies[anomalies['is_anomaly']][['date', 'correlation', 'anomaly_score']])

    # Display plot
    corr.plot()


if __name__ == '__main__':
    example_daily_correlation_radiation()
