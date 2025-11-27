"""
Fractional Kelly Position Sizer for Options Trading

Implements the Kelly Criterion with fractional sizing for risk management.
Based on research from Edward Thorp and academic literature on optimal betting.

Key formulas:
- Full Kelly: f* = (p * b - q) / b
  where p = P(win), q = P(loss), b = win/loss ratio

- Fractional Kelly: position = f* × fraction
  fraction = 0.25 gives ~1/81 chance of halving account

- Risk of Ruin: P(ruin) ≈ ((1-edge)/edge)^(bankroll/unit)
  where edge = p - q/b

Pipeline: RANKER -> META-LABELER -> KELLY SIZER -> EXECUTION

Author: Generated with Claude Code
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import logging
import joblib
import argparse
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class KellyConfig:
    """Configuration for Kelly position sizing."""
    # Kelly fraction (0.25 = quarter Kelly, recommended for trading)
    kelly_fraction: float = 0.25

    # Historical win/loss ratio estimation
    default_win_loss_ratio: float = 1.5  # If no historical data
    use_rolling_ratio: bool = True
    rolling_window: int = 100  # Trades to estimate win/loss ratio

    # Position limits
    min_position_pct: float = 0.005  # 0.5% minimum (lowered to allow smaller positions)
    max_position_pct: float = 0.05   # 5% maximum per position (more positions, smaller each)
    max_portfolio_risk: float = 0.50  # 50% max total risk (increased for more trades)

    # Risk of ruin constraints
    max_risk_of_ruin: float = 0.02   # 2% max probability of ruin (relaxed)
    ruin_threshold: float = 0.50     # Define ruin as losing 50% of capital

    # Probability thresholds
    min_prob_to_trade: float = 0.45  # Trade if P(win) >= 45% (lowered from 52%)
    high_conviction_threshold: float = 0.55  # Increase size above 55% (lowered from 65%)


@dataclass(frozen=True)
class CorrelationConfig:
    """Configuration for correlation-aware position adjustment (Layer 5).

    Standard Kelly assumes independent bets. Options on correlated underlyings
    violate this assumption. This config controls regime-aware correlation
    adjustment.
    """
    # VIX regime thresholds
    vix_low_threshold: float = 15.0   # Low VIX = use longer lookback
    vix_high_threshold: float = 25.0  # High VIX = assume stress correlation

    # Correlation estimation
    low_vix_lookback: int = 126      # ~6 months for low VIX
    high_vix_lookback: int = 21      # ~1 month for medium VIX
    stress_correlation: float = 0.70  # Assumed correlation in stress regimes
    default_correlation: float = 0.40 # Default when no history available

    # Clustering
    cluster_correlation_threshold: float = 0.60  # Positions > this are clustered
    max_cluster_allocation: float = 0.40  # Max 40% in any correlated cluster

    # Enable/disable
    enabled: bool = True


# =============================================================================
# Correlation Adjustment (Layer 5)
# =============================================================================

class CorrelationAdjuster:
    """
    Adjusts position sizes to account for correlation between positions.

    Standard Kelly assumes independent bets. Options on correlated underlyings
    violate this assumption. We use regime-conditional correlation matrices
    and hierarchical sizing to handle this.

    Based on research from López de Prado and portfolio optimization literature.
    """

    def __init__(self, config: CorrelationConfig = None):
        self.config = config or CorrelationConfig()
        self.correlation_cache: Dict[str, pd.DataFrame] = {}

    def get_regime_correlation(
        self,
        underlyings: List[str],
        vix_level: float,
        returns_history: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get correlation matrix based on current VIX regime.

        Low VIX (<15): Use 6-month historical correlation (stable markets)
        Medium VIX (15-25): Use 1-month correlation (more recent = more relevant)
        High VIX (>25): Assume stress correlation (~0.7 across all positions)

        Args:
            underlyings: List of underlying symbols
            vix_level: Current VIX level for regime detection
            returns_history: DataFrame of historical returns by underlying

        Returns:
            Correlation matrix as DataFrame
        """
        n = len(underlyings)

        if n == 0:
            return pd.DataFrame()

        if n == 1:
            return pd.DataFrame([[1.0]], index=underlyings, columns=underlyings)

        # Stress regime: assume high correlation across all positions
        if vix_level >= self.config.vix_high_threshold:
            logger.info(f"VIX {vix_level:.1f} >= {self.config.vix_high_threshold} - using stress correlation {self.config.stress_correlation}")
            corr = np.full((n, n), self.config.stress_correlation)
            np.fill_diagonal(corr, 1.0)
            return pd.DataFrame(corr, index=underlyings, columns=underlyings)

        # No history available: use default correlation
        if returns_history is None or returns_history.empty:
            corr = np.full((n, n), self.config.default_correlation)
            np.fill_diagonal(corr, 1.0)
            return pd.DataFrame(corr, index=underlyings, columns=underlyings)

        # Determine lookback based on VIX regime
        if vix_level < self.config.vix_low_threshold:
            lookback = self.config.low_vix_lookback
            regime = "low"
        else:
            lookback = self.config.high_vix_lookback
            regime = "medium"

        logger.debug(f"VIX {vix_level:.1f} - {regime} regime, using {lookback} day lookback")

        # Compute correlation from returns history
        recent = returns_history.tail(lookback)
        available = [u for u in underlyings if u in recent.columns]

        if len(available) < 2:
            corr = np.full((n, n), self.config.default_correlation)
            np.fill_diagonal(corr, 1.0)
            return pd.DataFrame(corr, index=underlyings, columns=underlyings)

        # Calculate correlation for available underlyings
        corr_available = recent[available].corr().fillna(self.config.default_correlation)

        # Build full correlation matrix
        corr = pd.DataFrame(
            self.config.default_correlation,
            index=underlyings,
            columns=underlyings
        )
        np.fill_diagonal(corr.values, 1.0)

        # Fill in available correlations
        for u1 in available:
            for u2 in available:
                if u1 in corr.index and u2 in corr.columns:
                    corr.loc[u1, u2] = corr_available.loc[u1, u2]

        return corr

    def cluster_positions(
        self,
        correlation_matrix: pd.DataFrame
    ) -> Dict[int, List[str]]:
        """
        Cluster positions by correlation using hierarchical clustering.

        Positions with correlation > threshold are grouped together.
        This allows us to cap exposure to correlated groups.

        Args:
            correlation_matrix: Square correlation matrix

        Returns:
            Dict mapping cluster_id -> list of underlying symbols
        """
        if len(correlation_matrix) <= 1:
            return {0: list(correlation_matrix.index)}

        # Convert correlation to distance (high corr = low distance)
        distance = 1 - correlation_matrix.abs()
        np.fill_diagonal(distance.values, 0)

        try:
            # Condensed distance matrix for linkage
            condensed = squareform(distance.values, checks=False)

            # Hierarchical clustering
            Z = linkage(condensed, method='average')

            # Cut tree at threshold
            threshold = 1 - self.config.cluster_correlation_threshold
            clusters = fcluster(Z, t=threshold, criterion='distance')

        except Exception as e:
            logger.warning(f"Clustering failed ({e}), treating each position separately")
            clusters = np.arange(len(correlation_matrix))

        # Group by cluster
        cluster_groups = {}
        for underlying, cluster_id in zip(correlation_matrix.index, clusters):
            cluster_id = int(cluster_id)
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(underlying)

        return cluster_groups

    def adjust_sizes(
        self,
        positions_df: pd.DataFrame,
        vix_level: float = 20.0,
        returns_history: Optional[pd.DataFrame] = None,
        size_column: str = 'position_size',
        underlying_column: str = 'underlying'
    ) -> pd.DataFrame:
        """
        Adjust position sizes to respect correlation constraints.

        Pipeline:
        1. Get regime-appropriate correlation matrix
        2. Cluster correlated positions
        3. Cap total allocation per cluster
        4. Normalize if exceeds portfolio limit

        Args:
            positions_df: DataFrame with positions and sizes
            vix_level: Current VIX level for regime detection
            returns_history: Historical returns by underlying (optional)
            size_column: Column name for position sizes
            underlying_column: Column name for underlying symbol

        Returns:
            DataFrame with adjusted sizes in 'adjusted_size' column
        """
        if not self.config.enabled:
            positions_df['adjusted_size'] = positions_df[size_column]
            return positions_df

        result = positions_df.copy()

        # Filter to positions with size > 0
        active_mask = result[size_column] > 0
        if active_mask.sum() <= 1:
            result['adjusted_size'] = result[size_column]
            return result

        # Get unique underlyings
        if underlying_column not in result.columns:
            logger.warning(f"No {underlying_column} column - skipping correlation adjustment")
            result['adjusted_size'] = result[size_column]
            return result

        underlyings = result.loc[active_mask, underlying_column].unique().tolist()

        if len(underlyings) <= 1:
            result['adjusted_size'] = result[size_column]
            return result

        # Get regime-appropriate correlation matrix
        corr = self.get_regime_correlation(underlyings, vix_level, returns_history)

        # Cluster correlated positions
        clusters = self.cluster_positions(corr)

        logger.info(f"Correlation adjustment: {len(underlyings)} underlyings -> {len(clusters)} clusters (VIX={vix_level:.1f})")

        # Initialize adjusted sizes
        result['adjusted_size'] = result[size_column].copy()

        # Apply cluster-level caps
        for cluster_id, cluster_underlyings in clusters.items():
            cluster_mask = (
                result[underlying_column].isin(cluster_underlyings) &
                active_mask
            )
            cluster_total = result.loc[cluster_mask, size_column].sum()

            if cluster_total > self.config.max_cluster_allocation:
                # Scale down proportionally within cluster
                scale = self.config.max_cluster_allocation / cluster_total
                result.loc[cluster_mask, 'adjusted_size'] = (
                    result.loc[cluster_mask, size_column] * scale
                )
                logger.info(
                    f"Cluster {cluster_id} ({cluster_underlyings}): "
                    f"scaled {cluster_total:.1%} -> {self.config.max_cluster_allocation:.1%}"
                )

        # Ensure total doesn't exceed 100%
        total = result['adjusted_size'].sum()
        if total > 1.0:
            result['adjusted_size'] = result['adjusted_size'] / total
            logger.info(f"Total allocation {total:.1%} -> normalized to 100%")

        return result


# =============================================================================
# Kelly Calculations
# =============================================================================

def confidence_score(prob: float) -> float:
    """
    Calculate confidence based on probability distance from 0.5.

    When prob is near 0.5, model is uncertain and we should reduce size.
    When prob is near 0 or 1, model is confident.

    Returns value in [0, 1] where:
    - 0 at prob=0.5 (maximum uncertainty)
    - 1 at prob=0 or prob=1 (maximum confidence)
    """
    # Entropy-based: 1 - 4*p*(1-p)
    # At p=0.5: 1 - 4*0.25 = 0
    # At p=0 or 1: 1 - 0 = 1
    return 1 - 4 * prob * (1 - prob)


def dynamic_win_loss_ratio(
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    expected_move: float = 0.02,  # 2% expected move
    days_to_hold: int = 5,
    option_price: float = 1.0
) -> float:
    """
    Calculate win/loss ratio from Greeks (more principled than historical avg).

    Expected profit ≈ delta × move + 0.5 × gamma × move² + theta × time - vega × IV_crush
    Expected loss = premium paid (for long options)

    Args:
        delta: Option delta
        gamma: Option gamma
        theta: Option theta (negative for long)
        vega: Option vega
        expected_move: Expected underlying move (fraction)
        days_to_hold: Days to hold position
        option_price: Current option price

    Returns:
        Estimated win/loss ratio
    """
    if option_price <= 0:
        return 1.5  # Default fallback

    # Expected profit from favorable move
    # Assume IV crush of 10% on average after entry
    iv_crush = 0.10

    expected_pnl = (
        abs(delta) * expected_move +
        0.5 * gamma * (expected_move ** 2) +
        theta * days_to_hold -
        vega * iv_crush
    )

    # Expected loss = premium (for long options, capped at premium)
    expected_loss = min(option_price, abs(theta * days_to_hold * 2))
    expected_loss = max(expected_loss, option_price * 0.2)  # At least 20% loss

    if expected_loss <= 0:
        return 1.5

    ratio = abs(expected_pnl) / expected_loss
    # Clamp to reasonable range
    return np.clip(ratio, 0.5, 5.0)


def kelly_criterion(
    prob_win: float,
    win_loss_ratio: float,
    fraction: float = 1.0
) -> float:
    """
    Calculate Kelly optimal bet fraction.

    Args:
        prob_win: Probability of winning (0-1)
        win_loss_ratio: Average win / Average loss
        fraction: Kelly fraction (0.25 = quarter Kelly)

    Returns:
        Optimal position size as fraction of bankroll

    Formula:
        f* = (p * b - q) / b
        where p = prob_win, q = 1-p, b = win_loss_ratio
    """
    if prob_win <= 0 or prob_win >= 1:
        return 0.0

    if win_loss_ratio <= 0:
        return 0.0

    q = 1 - prob_win
    b = win_loss_ratio

    # Kelly formula
    kelly = (prob_win * b - q) / b

    # Apply fraction and ensure non-negative
    return max(0.0, kelly * fraction)


def risk_of_ruin(
    prob_win: float,
    win_loss_ratio: float,
    position_size: float,
    ruin_threshold: float = 0.5
) -> float:
    """
    Estimate probability of reaching ruin threshold.

    Uses simplified formula: P(ruin) ≈ ((1-edge)/edge)^N
    where N = 1/position_size (number of bets to reach ruin)

    Args:
        prob_win: Probability of winning
        win_loss_ratio: Average win / Average loss
        position_size: Fraction of bankroll per bet
        ruin_threshold: What fraction loss constitutes ruin

    Returns:
        Estimated probability of ruin
    """
    if position_size <= 0:
        return 0.0

    # Calculate edge
    q = 1 - prob_win
    expected_return = prob_win * win_loss_ratio - q

    if expected_return <= 0:
        return 1.0  # Negative edge = eventual ruin

    # Number of consecutive losses to reach ruin
    n_losses_to_ruin = np.log(1 - ruin_threshold) / np.log(1 - position_size)

    # Probability of n consecutive losses
    prob_n_losses = q ** n_losses_to_ruin

    # This is simplified - actual risk of ruin is more complex
    # but provides a conservative upper bound
    return min(1.0, prob_n_losses * 2)  # 2x safety factor


def optimal_fraction_for_risk(
    prob_win: float,
    win_loss_ratio: float,
    target_risk_of_ruin: float = 0.01,
    ruin_threshold: float = 0.5
) -> float:
    """
    Find Kelly fraction that achieves target risk of ruin.

    Binary search for the fraction that gives desired risk level.

    Args:
        prob_win: Probability of winning
        win_loss_ratio: Average win / Average loss
        target_risk_of_ruin: Desired maximum risk of ruin
        ruin_threshold: What fraction loss constitutes ruin

    Returns:
        Kelly fraction to use
    """
    # Binary search for optimal fraction
    low, high = 0.01, 1.0

    for _ in range(50):  # Max iterations
        mid = (low + high) / 2
        position = kelly_criterion(prob_win, win_loss_ratio, mid)
        ror = risk_of_ruin(prob_win, win_loss_ratio, position, ruin_threshold)

        if ror > target_risk_of_ruin:
            high = mid
        else:
            low = mid

        if high - low < 0.001:
            break

    return low  # Conservative choice


def confidence_adjusted_kelly(
    prob_win: float,
    win_loss_ratio: float,
    fraction: float = 0.25,
    use_confidence: bool = True
) -> float:
    """
    Kelly criterion with confidence adjustment.

    When model is uncertain (prob near 0.5), reduce position size
    beyond what Kelly prescribes.

    Args:
        prob_win: Probability of winning
        win_loss_ratio: Win/loss ratio
        fraction: Kelly fraction
        use_confidence: Whether to apply confidence adjustment

    Returns:
        Confidence-adjusted position size
    """
    base_kelly = kelly_criterion(prob_win, win_loss_ratio, fraction)

    if not use_confidence or base_kelly <= 0:
        return base_kelly

    # Get confidence (0 at p=0.5, 1 at p=0 or 1)
    conf = confidence_score(prob_win)

    # Apply confidence adjustment
    # At p=0.55, conf ≈ 0.19, so size is reduced by ~80%
    # At p=0.70, conf ≈ 0.64, so size is reduced by ~36%
    return base_kelly * conf


def correlation_penalty(
    positions: pd.DataFrame,
    correlation_matrix: Optional[np.ndarray] = None,
    underlying_column: str = 'underlying',
    correlation_threshold: float = 0.6
) -> float:
    """
    Calculate correlation penalty for portfolio of positions.

    If positions are highly correlated, treat them as one big bet
    and reduce total allocation.

    Args:
        positions: DataFrame with position info
        correlation_matrix: Optional correlation matrix
        underlying_column: Column identifying underlying asset
        correlation_threshold: Threshold for "highly correlated"

    Returns:
        Penalty factor in (0, 1] to multiply position sizes by
    """
    if len(positions) <= 1:
        return 1.0

    # If we have underlying info, estimate correlation from same underlying
    if underlying_column in positions.columns:
        underlyings = positions[underlying_column].values
        n = len(underlyings)

        # Count pairs with same underlying
        same_underlying_pairs = 0
        total_pairs = n * (n - 1) / 2

        for i in range(n):
            for j in range(i + 1, n):
                if underlyings[i] == underlyings[j]:
                    same_underlying_pairs += 1

        if total_pairs > 0:
            concentration = same_underlying_pairs / total_pairs
        else:
            concentration = 0

        # Penalty: if all same underlying, penalty = 1/n (treat as one bet)
        # If all different, penalty = 1 (no adjustment)
        penalty = 1 / (1 + concentration * (n - 1))
        return penalty

    # If correlation matrix provided, use it
    if correlation_matrix is not None:
        avg_corr = np.mean(np.abs(correlation_matrix))
        n = len(positions)
        penalty = 1 / (1 + avg_corr * (n - 1))
        return penalty

    # Default: no penalty
    return 1.0


def hierarchical_kelly(
    kelly_sizes: np.ndarray,
    groups: np.ndarray,
    max_group_allocation: float = 0.15
) -> np.ndarray:
    """
    Apply hierarchical Kelly sizing within correlation clusters.

    Caps exposure to correlated groups (e.g., all tech stocks).

    Args:
        kelly_sizes: Raw Kelly sizes for each position
        groups: Group/cluster ID for each position
        max_group_allocation: Maximum total allocation per group

    Returns:
        Adjusted Kelly sizes
    """
    adjusted = kelly_sizes.copy()
    unique_groups = np.unique(groups)

    for group in unique_groups:
        mask = groups == group
        group_total = kelly_sizes[mask].sum()

        if group_total > max_group_allocation:
            # Scale down proportionally
            scale = max_group_allocation / group_total
            adjusted[mask] = kelly_sizes[mask] * scale

    return adjusted


# =============================================================================
# Position Sizer Class
# =============================================================================

class KellySizer:
    """
    Fractional Kelly position sizer for options trading.

    Takes probability estimates from meta-labeler and outputs
    position sizes that maximize long-term growth while
    controlling risk of ruin.

    Includes Layer 5 correlation adjustment for correlated positions.
    """

    def __init__(
        self,
        config: KellyConfig,
        correlation_config: Optional[CorrelationConfig] = None
    ):
        self.config = config
        self.historical_trades: List[Dict] = []
        self.win_loss_ratio = config.default_win_loss_ratio
        self.correlation_adjuster = CorrelationAdjuster(
            correlation_config or CorrelationConfig()
        )

    def update_win_loss_ratio(self, trades: List[Dict]) -> float:
        """
        Update win/loss ratio from historical trades.

        Args:
            trades: List of trade dicts with 'pnl' key

        Returns:
            Updated win/loss ratio
        """
        if not trades:
            return self.config.default_win_loss_ratio

        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]

        if not wins or not losses:
            return self.config.default_win_loss_ratio

        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        if avg_loss <= 0:
            return self.config.default_win_loss_ratio

        self.win_loss_ratio = avg_win / avg_loss
        logger.info(f"Updated win/loss ratio: {self.win_loss_ratio:.2f}")

        return self.win_loss_ratio

    def size_position(
        self,
        prob_profit: float,
        win_loss_ratio: Optional[float] = None,
        current_portfolio_risk: float = 0.0,
        use_confidence_adjustment: bool = True,
        greeks: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate position size for a single trade.

        Args:
            prob_profit: Calibrated probability of profit from meta-labeler
            win_loss_ratio: Override win/loss ratio (uses historical if None)
            current_portfolio_risk: Current total portfolio risk exposure
            use_confidence_adjustment: Apply confidence penalty for uncertain predictions
            greeks: Optional dict with delta, gamma, theta, vega for dynamic W/L ratio

        Returns:
            Dict with position_size, kelly_full, risk_of_ruin, etc.
        """
        # Calculate win/loss ratio
        if win_loss_ratio is not None:
            wl_ratio = win_loss_ratio
        elif greeks is not None:
            # Use Greeks for dynamic W/L ratio (more principled)
            wl_ratio = dynamic_win_loss_ratio(
                delta=greeks.get('delta', 0.5),
                gamma=greeks.get('gamma', 0.01),
                theta=greeks.get('theta', -0.05),
                vega=greeks.get('vega', 0.1),
                option_price=greeks.get('price', 1.0)
            )
        else:
            wl_ratio = self.win_loss_ratio

        # Check minimum probability threshold
        if prob_profit < self.config.min_prob_to_trade:
            return {
                'position_size': 0.0,
                'kelly_full': 0.0,
                'kelly_fraction_used': self.config.kelly_fraction,
                'confidence': confidence_score(prob_profit),
                'risk_of_ruin': 0.0,
                'skip_reason': f'prob {prob_profit:.2f} < min {self.config.min_prob_to_trade:.2f}'
            }

        # Calculate full Kelly
        kelly_full = kelly_criterion(prob_profit, wl_ratio, fraction=1.0)

        if kelly_full <= 0:
            return {
                'position_size': 0.0,
                'kelly_full': kelly_full,
                'kelly_fraction_used': self.config.kelly_fraction,
                'confidence': confidence_score(prob_profit),
                'risk_of_ruin': 0.0,
                'skip_reason': 'negative Kelly (no edge)'
            }

        # Apply fractional Kelly
        fraction = self.config.kelly_fraction

        # Adjust fraction for high conviction trades
        if prob_profit >= self.config.high_conviction_threshold:
            fraction = min(0.5, fraction * 1.5)  # Up to 50% more for high conviction

        # Calculate confidence score
        conf = confidence_score(prob_profit)

        # Apply confidence adjustment (reduce size when uncertain)
        if use_confidence_adjustment:
            position_size = kelly_full * fraction * conf
        else:
            position_size = kelly_full * fraction

        # Apply position limits
        position_size = max(position_size, self.config.min_position_pct) if position_size > 0 else 0
        position_size = min(position_size, self.config.max_position_pct)

        # Check portfolio risk limit
        remaining_risk = self.config.max_portfolio_risk - current_portfolio_risk
        if position_size > remaining_risk:
            position_size = max(0, remaining_risk)

        # Calculate risk of ruin
        ror = risk_of_ruin(
            prob_profit, wl_ratio, position_size,
            self.config.ruin_threshold
        )

        # If risk of ruin too high, reduce position
        if ror > self.config.max_risk_of_ruin:
            # Find safe fraction
            safe_fraction = optimal_fraction_for_risk(
                prob_profit, wl_ratio,
                self.config.max_risk_of_ruin,
                self.config.ruin_threshold
            )
            position_size = kelly_full * safe_fraction
            if use_confidence_adjustment:
                position_size *= conf
            ror = risk_of_ruin(
                prob_profit, wl_ratio, position_size,
                self.config.ruin_threshold
            )
            fraction = safe_fraction

        return {
            'position_size': position_size,
            'kelly_full': kelly_full,
            'kelly_fraction_used': fraction,
            'confidence': conf,
            'risk_of_ruin': ror,
            'prob_profit': prob_profit,
            'win_loss_ratio': wl_ratio,
            'skip_reason': None
        }

    def size_portfolio(
        self,
        opportunities: pd.DataFrame,
        prob_column: str = 'prob_profit',
        max_positions: int = 10,
        apply_correlation_adjustment: bool = True,
        underlying_column: str = 'underlying',
        vix_level: float = 20.0,
        returns_history: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Size positions for multiple opportunities.

        Allocates capital across opportunities respecting:
        - Individual position limits
        - Portfolio risk limits
        - Diversification (via Layer 5 correlation adjustment)
        - Confidence adjustment

        Args:
            opportunities: DataFrame with prob_profit column
            prob_column: Column name for probability
            max_positions: Maximum number of positions
            apply_correlation_adjustment: Apply Layer 5 correlation adjustment
            underlying_column: Column identifying underlying asset
            vix_level: Current VIX level for regime-based correlation
            returns_history: Historical returns by underlying for correlation estimation

        Returns:
            DataFrame with position sizes added
        """
        if opportunities.empty:
            return opportunities.assign(position_size=0.0)

        df = opportunities.copy()

        # Sort by probability (best opportunities first)
        df = df.sort_values(prob_column, ascending=False)

        # Check for Greeks columns for dynamic W/L ratio
        has_greeks = all(col in df.columns for col in ['delta', 'gamma', 'theta', 'vega'])

        # Size each position
        sizes = []
        current_risk = 0.0

        for idx, row in df.iterrows():
            if len(sizes) >= max_positions:
                sizing = {
                    'position_size': 0.0,
                    'kelly_full': 0.0,
                    'kelly_fraction_used': self.config.kelly_fraction,
                    'confidence': 0.0,
                    'risk_of_ruin': 0.0,
                    'win_loss_ratio': self.win_loss_ratio,
                    'skip_reason': 'max_positions'
                }
            else:
                # Build Greeks dict if available
                greeks = None
                if has_greeks:
                    price_col = 'last' if 'last' in df.columns else 'price'
                    greeks = {
                        'delta': row.get('delta', 0.5),
                        'gamma': row.get('gamma', 0.01),
                        'theta': row.get('theta', -0.05),
                        'vega': row.get('vega', 0.1),
                        'price': row.get(price_col, 1.0) if price_col in row else 1.0
                    }

                sizing = self.size_position(
                    row[prob_column],
                    current_portfolio_risk=current_risk,
                    greeks=greeks
                )
                current_risk += sizing['position_size']

            sizes.append(sizing)

        # Add sizing columns to dataframe
        df['position_size'] = [s['position_size'] for s in sizes]
        df['kelly_full'] = [s.get('kelly_full', 0.0) for s in sizes]
        df['kelly_fraction'] = [s.get('kelly_fraction_used', 0.0) for s in sizes]
        df['confidence'] = [s.get('confidence', 0.0) for s in sizes]
        df['risk_of_ruin'] = [s.get('risk_of_ruin', 0.0) for s in sizes]
        df['win_loss_ratio'] = [s.get('win_loss_ratio', 0.0) for s in sizes]
        df['skip_reason'] = [s.get('skip_reason') for s in sizes]

        # Apply Layer 5 correlation adjustment if requested
        if apply_correlation_adjustment and underlying_column in df.columns:
            n_active = (df['position_size'] > 0).sum()
            if n_active > 1:
                # Use the new CorrelationAdjuster with regime-aware correlation
                df = self.correlation_adjuster.adjust_sizes(
                    df,
                    vix_level=vix_level,
                    returns_history=returns_history,
                    size_column='position_size',
                    underlying_column=underlying_column
                )
                # Use adjusted_size if available, otherwise keep position_size
                if 'adjusted_size' in df.columns:
                    df['position_size'] = df['adjusted_size']
            else:
                df['adjusted_size'] = df['position_size']
        else:
            df['adjusted_size'] = df['position_size']

        # Log summary
        total_risk = df['position_size'].sum()
        n_trades = (df['position_size'] > 0).sum()
        avg_confidence = df[df['position_size'] > 0]['confidence'].mean() if n_trades > 0 else 0
        logger.info(f"Sized {n_trades}/{len(df)} positions, total risk: {total_risk:.1%}, avg confidence: {avg_confidence:.2f}")

        return df

    def calculate_expected_growth(
        self,
        prob_profit: float,
        win_loss_ratio: float,
        position_size: float
    ) -> float:
        """
        Calculate expected log growth rate.

        This is what Kelly maximizes.

        Args:
            prob_profit: Probability of winning
            win_loss_ratio: Win/loss ratio
            position_size: Position size as fraction

        Returns:
            Expected log growth per bet
        """
        if position_size <= 0:
            return 0.0

        q = 1 - prob_profit

        # Expected log growth
        # E[log(1 + f*X)] where X is return
        growth = (
            prob_profit * np.log(1 + position_size * win_loss_ratio) +
            q * np.log(1 - position_size)
        )

        return growth

    def simulate_trajectory(
        self,
        prob_profit: float,
        win_loss_ratio: float,
        position_size: float,
        n_trades: int = 1000,
        n_simulations: int = 1000,
        initial_capital: float = 1.0
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation of portfolio trajectory.

        Args:
            prob_profit: Probability of winning each trade
            win_loss_ratio: Win/loss ratio
            position_size: Position size as fraction
            n_trades: Number of trades to simulate
            n_simulations: Number of simulation paths
            initial_capital: Starting capital

        Returns:
            Dict with statistics: final_mean, final_median, max_drawdown, prob_ruin
        """
        np.random.seed(42)

        # Generate random outcomes
        outcomes = np.random.random((n_simulations, n_trades)) < prob_profit

        # Calculate returns for each outcome
        returns = np.where(
            outcomes,
            position_size * win_loss_ratio,  # Win
            -position_size                     # Loss
        )

        # Calculate cumulative wealth
        wealth = initial_capital * np.cumprod(1 + returns, axis=1)

        # Calculate statistics
        final_wealth = wealth[:, -1]

        # Max drawdown
        running_max = np.maximum.accumulate(wealth, axis=1)
        drawdowns = (running_max - wealth) / running_max
        max_drawdowns = np.max(drawdowns, axis=1)

        # Probability of ruin (losing > threshold)
        ruin_count = np.sum(final_wealth < initial_capital * (1 - self.config.ruin_threshold))
        prob_ruin = ruin_count / n_simulations

        return {
            'final_mean': np.mean(final_wealth),
            'final_median': np.median(final_wealth),
            'final_std': np.std(final_wealth),
            'max_drawdown_mean': np.mean(max_drawdowns),
            'max_drawdown_95': np.percentile(max_drawdowns, 95),
            'prob_ruin': prob_ruin,
            'cagr': (np.median(final_wealth) ** (1/n_trades) - 1) * 252,  # Annualized
        }


# =============================================================================
# Integration with Meta-Labeler
# =============================================================================

def size_from_meta_labeler(
    meta_labeler_path: str,
    data_df: pd.DataFrame,
    kelly_config: KellyConfig,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Full pipeline: Load meta-labeler predictions and size positions.

    Args:
        meta_labeler_path: Path to saved meta-labeler model
        data_df: DataFrame with features for prediction
        kelly_config: Kelly sizer configuration
        output_path: Optional path to save results

    Returns:
        DataFrame with position sizes
    """
    # Load meta-labeler
    artifacts = joblib.load(meta_labeler_path)

    # Create temporary meta-labeler for prediction
    from prod_meta_labeler import MetaLabeler, MetaLabelerConfig

    config = artifacts.get('config', MetaLabelerConfig())
    model = MetaLabeler(config)
    model.classifier = artifacts['classifier']
    model.calibrator = artifacts['calibrator']
    model.scaler = artifacts['scaler']
    model.imputer = artifacts['imputer']
    model.feature_names = artifacts['feature_names']

    # Get probability predictions
    logger.info("Generating probability predictions...")
    data_df['prob_profit'] = model.predict_proba(data_df)

    # Size positions
    sizer = KellySizer(kelly_config)
    sized_df = sizer.size_portfolio(data_df, prob_column='prob_profit')

    # Summary statistics
    logger.info("\n=== Position Sizing Summary ===")
    logger.info(f"Total opportunities: {len(sized_df)}")
    logger.info(f"Positions to take: {(sized_df['position_size'] > 0).sum()}")
    logger.info(f"Avg probability: {sized_df['prob_profit'].mean():.2%}")
    logger.info(f"Avg position size: {sized_df[sized_df['position_size'] > 0]['position_size'].mean():.2%}")
    logger.info(f"Total portfolio risk: {sized_df['position_size'].sum():.2%}")

    if output_path:
        sized_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")

    return sized_df


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kelly Position Sizer for Options Trading"
    )
    parser.add_argument(
        '--meta-labeler',
        type=str,
        help='Path to trained meta-labeler model'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to data CSV for sizing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='sized_positions.csv',
        help='Output path for sized positions'
    )
    parser.add_argument(
        '--kelly-fraction',
        type=float,
        default=0.25,
        help='Kelly fraction (0.25 = quarter Kelly)'
    )
    parser.add_argument(
        '--max-position',
        type=float,
        default=0.10,
        help='Maximum position size as fraction'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run Monte Carlo simulation'
    )

    args = parser.parse_args()

    config = KellyConfig(
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position,
    )

    if args.simulate:
        # Run simulation to demonstrate Kelly properties
        logger.info("Running Kelly Criterion demonstration...")

        sizer = KellySizer(config)

        # Example scenarios
        scenarios = [
            {'prob': 0.55, 'wl_ratio': 1.5, 'name': 'Modest edge'},
            {'prob': 0.60, 'wl_ratio': 1.5, 'name': 'Good edge'},
            {'prob': 0.65, 'wl_ratio': 2.0, 'name': 'Strong edge'},
            {'prob': 0.70, 'wl_ratio': 2.0, 'name': 'Excellent edge'},
        ]

        print("\n=== Kelly Criterion Analysis ===\n")
        print(f"{'Scenario':<20} {'Prob':<8} {'W/L':<8} {'Full Kelly':<12} "
              f"{'1/4 Kelly':<12} {'RoR (1/4)':<10}")
        print("-" * 80)

        for s in scenarios:
            full = kelly_criterion(s['prob'], s['wl_ratio'], 1.0)
            quarter = kelly_criterion(s['prob'], s['wl_ratio'], 0.25)
            ror = risk_of_ruin(s['prob'], s['wl_ratio'], quarter, 0.5)

            print(f"{s['name']:<20} {s['prob']:<8.0%} {s['wl_ratio']:<8.1f} "
                  f"{full:<12.1%} {quarter:<12.1%} {ror:<10.3%}")

        # Detailed simulation for one scenario
        print("\n=== Monte Carlo Simulation (1000 trades, 1000 paths) ===")
        print("Scenario: P(win)=60%, W/L=1.5, Quarter Kelly\n")

        sizing = sizer.size_position(0.60, 1.5)
        sim = sizer.simulate_trajectory(
            0.60, 1.5, sizing['position_size'],
            n_trades=1000, n_simulations=1000
        )

        print(f"Position size: {sizing['position_size']:.1%}")
        print(f"Final wealth (mean): {sim['final_mean']:.2f}x")
        print(f"Final wealth (median): {sim['final_median']:.2f}x")
        print(f"Max drawdown (95th %): {sim['max_drawdown_95']:.1%}")
        print(f"Probability of ruin: {sim['prob_ruin']:.2%}")
        print(f"Annualized CAGR: {sim['cagr']:.1%}")

    elif args.meta_labeler and args.data:
        # Size positions from data
        data_df = pd.read_csv(args.data)
        size_from_meta_labeler(
            args.meta_labeler,
            data_df,
            config,
            args.output
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
