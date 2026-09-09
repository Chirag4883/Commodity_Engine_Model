from __future__ import annotations

"""
Level 4 — Factor Cross-Sectional Orthogonalizer

Purpose
-------
Convert Module-1 raw supply-chain scores into a cross-sectional residual-alpha
vector that is orthogonal to systematic style and sector exposures.

Input columns
-------------
raw_score
log_market_cap
momentum_12m
price_to_book
sector

Model
-----
raw_score_i
    = alpha
    + beta_size * size_i
    + beta_mom * momentum_i
    + beta_value * value_i
    + sum_k beta_sector,k * D_sector,k,i
    + epsilon_i

Processing
----------
1. Validate a complete cross-section.
2. Standardize continuous risk factors for numerical conditioning.
3. Run OLS.
4. Extract residual epsilon.
5. Winsorize residual at +/- 3 population standard deviations.
6. Re-project the winsorized residual onto the orthogonal complement of the
   SAME factor design matrix.

   This step is necessary because clipping is nonlinear and can otherwise
   reintroduce factor correlations.

7. Standardize the corrected residual to:
       mean = 0
       population standard deviation = 1

The output is the Module-1 cross-sectional alpha Z-score vector.

Important
---------
This module does NOT select stocks.

Every valid input security receives exactly one numerical alpha score.
"""

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NeutralizationError(Exception):
    """Base exception for factor-neutralization failures."""


class NeutralizationInputError(NeutralizationError):
    """Invalid cross-sectional input."""


class RankDeficientDesignError(NeutralizationError):
    """Risk-factor matrix is not full column rank."""


class IllConditionedDesignError(NeutralizationError):
    """Risk-factor design matrix is numerically unstable."""


class DegenerateAlphaError(NeutralizationError):
    """No usable idiosyncratic variation remains."""


class NeutralityViolationError(NeutralizationError):
    """Final alpha fails the configured neutrality tolerance."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeutralizerConfig:
    raw_score_column: str = "raw_score"

    size_column: str = "log_market_cap"

    momentum_column: str = "momentum_12m"

    value_column: str = "price_to_book"

    sector_column: str = "sector"

    winsor_sigma: float = 3.0

    min_observations: int = 30

    max_condition_number: float = 1.0e10

    neutrality_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.winsor_sigma)
            or self.winsor_sigma <= 0.0
        ):
            raise ValueError(
                "winsor_sigma must be finite and positive"
            )

        if self.min_observations < 5:
            raise ValueError(
                "min_observations must be at least 5"
            )

        if (
            not math.isfinite(self.max_condition_number)
            or self.max_condition_number <= 1.0
        ):
            raise ValueError(
                "max_condition_number must be finite and > 1"
            )

        if (
            not math.isfinite(self.neutrality_tolerance)
            or self.neutrality_tolerance <= 0.0
        ):
            raise ValueError(
                "neutrality_tolerance must be finite and positive"
            )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorStandardization:
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class NeutralizationDiagnostics:
    n_observations: int

    n_design_columns: int

    design_rank: int

    condition_number: float

    residual_mean_before_winsor: float

    residual_std_before_winsor: float

    winsor_lower_bound: float

    winsor_upper_bound: float

    n_winsorized_lower: int

    n_winsorized_upper: int

    max_abs_final_factor_correlation: float

    final_mean: float

    final_std: float


@dataclass(frozen=True, slots=True)
class NeutralizationResult:
    """
    Complete audit trail for one cross-sectional neutralization.
    """

    alpha_z: pd.Series

    raw_residual: pd.Series

    winsorized_residual: pd.Series

    orthogonalized_residual: pd.Series

    fitted_systematic_score: pd.Series

    coefficients: pd.Series

    continuous_factor_scaling: Mapping[
        str,
        FactorStandardization,
    ]

    sector_baseline: str

    factor_correlations: Mapping[str, float]

    diagnostics: NeutralizationDiagnostics

    def as_frame(self) -> pd.DataFrame:
        """
        Convenient downstream representation.
        """
        return pd.DataFrame(
            {
                "fitted_systematic_score":
                    self.fitted_systematic_score,

                "raw_residual":
                    self.raw_residual,

                "winsorized_residual":
                    self.winsorized_residual,

                "orthogonalized_residual":
                    self.orthogonalized_residual,

                "alpha_z":
                    self.alpha_z,
            }
        )


# ---------------------------------------------------------------------------
# Neutralizer
# ---------------------------------------------------------------------------


class FactorNeutralizer:
    """
    Cross-sectional OLS residualization engine.
    """

    def __init__(
        self,
        config: Optional[NeutralizerConfig] = None,
    ) -> None:
        self.config = (
            config
            or NeutralizerConfig()
        )

    def neutralize(
        self,
        cross_section: pd.DataFrame,
    ) -> NeutralizationResult:
        """
        Neutralize one PIT cross-section.

        Parameters
        ----------
        cross_section:
            DataFrame indexed by security/ticker.

            Required columns:
                raw_score
                log_market_cap
                momentum_12m
                price_to_book
                sector

        Returns
        -------
        NeutralizationResult
            Full numerical audit trail plus final alpha Z-score array.
        """

        frame = self._validate_and_prepare(
            cross_section
        )

        (
            design,
            design_columns,
            scaling,
            baseline_sector,
        ) = self._build_design_matrix(
            frame
        )

        y = frame[
            self.config.raw_score_column
        ].to_numpy(
            dtype=float
        )

        n_obs, n_columns = (
            design.shape
        )

        rank = int(
            np.linalg.matrix_rank(
                design
            )
        )

        if rank != n_columns:
            raise RankDeficientDesignError(
                "Risk-factor design matrix is rank deficient: "
                f"rank={rank}, columns={n_columns}"
            )

        condition_number = float(
            np.linalg.cond(
                design
            )
        )

        if (
            not math.isfinite(
                condition_number
            )
            or condition_number
            > self.config.max_condition_number
        ):
            raise IllConditionedDesignError(
                "Risk-factor design matrix is ill-conditioned: "
                f"condition_number={condition_number:.6g}"
            )

        # ------------------------------------------------------------------
        # First-pass OLS
        # ------------------------------------------------------------------

        coefficients_array, _, _, _ = (
            np.linalg.lstsq(
                design,
                y,
                rcond=None,
            )
        )

        fitted = (
            design
            @ coefficients_array
        )

        raw_residual = (
            y
            - fitted
        )

        residual_mean = float(
            np.mean(
                raw_residual
            )
        )

        residual_std = float(
            np.std(
                raw_residual,
                ddof=0,
            )
        )

        if (
            not math.isfinite(
                residual_std
            )
            or residual_std
            <= np.finfo(float).eps
        ):
            raise DegenerateAlphaError(
                "OLS residual has no usable "
                "cross-sectional variation"
            )

        # ------------------------------------------------------------------
        # 3-sigma winsorization
        # ------------------------------------------------------------------

        lower_bound = (
            residual_mean
            - (
                self.config.winsor_sigma
                * residual_std
            )
        )

        upper_bound = (
            residual_mean
            + (
                self.config.winsor_sigma
                * residual_std
            )
        )

        n_lower = int(
            np.sum(
                raw_residual
                < lower_bound
            )
        )

        n_upper = int(
            np.sum(
                raw_residual
                > upper_bound
            )
        )

        winsorized = np.clip(
            raw_residual,
            lower_bound,
            upper_bound,
        )

        # ------------------------------------------------------------------
        # Re-orthogonalize after clipping
        # ------------------------------------------------------------------
        #
        # Winsorization is nonlinear.
        #
        # Even if:
        #
        #     X' epsilon = 0
        #
        # clipping epsilon can produce:
        #
        #     X' clip(epsilon) != 0
        #
        # Therefore remove any factor projection introduced by clipping.
        # ------------------------------------------------------------------

        post_winsor_beta, _, _, _ = (
            np.linalg.lstsq(
                design,
                winsorized,
                rcond=None,
            )
        )

        orthogonalized = (
            winsorized
            - (
                design
                @ post_winsor_beta
            )
        )

        orthogonalized_std = float(
            np.std(
                orthogonalized,
                ddof=0,
            )
        )

        if (
            not math.isfinite(
                orthogonalized_std
            )
            or orthogonalized_std
            <= np.finfo(float).eps
        ):
            raise DegenerateAlphaError(
                "No usable idiosyncratic "
                "variation remains after winsorization"
            )

        # ------------------------------------------------------------------
        # Final Z-score
        # ------------------------------------------------------------------

        orthogonalized_mean = float(
            np.mean(
                orthogonalized
            )
        )

        alpha_z = (
            orthogonalized
            - orthogonalized_mean
        ) / orthogonalized_std

        # One final floating-point centering operation.
        #
        # Division preserves orthogonality; centering is safe because the
        # design contains an intercept.
        alpha_z = (
            alpha_z
            - np.mean(
                alpha_z
            )
        )

        final_std = float(
            np.std(
                alpha_z,
                ddof=0,
            )
        )

        if (
            not math.isfinite(
                final_std
            )
            or final_std
            <= np.finfo(float).eps
        ):
            raise DegenerateAlphaError(
                "Final standardized alpha is degenerate"
            )

        alpha_z = (
            alpha_z
            / final_std
        )

        # ------------------------------------------------------------------
        # Correlation diagnostics
        # ------------------------------------------------------------------

        factor_correlations = (
            self._calculate_factor_correlations(
                frame=frame,
                alpha_z=alpha_z,
            )
        )

        max_abs_correlation = (
            max(
                (
                    abs(value)
                    for value
                    in factor_correlations.values()
                ),
                default=0.0,
            )
        )

        if (
            max_abs_correlation
            >= self.config.neutrality_tolerance
        ):
            raise NeutralityViolationError(
                "Final alpha failed neutrality gate: "
                f"max |r|={max_abs_correlation:.12g}, "
                f"tolerance="
                f"{self.config.neutrality_tolerance:.12g}"
            )

        index = (
            frame.index.copy()
        )

        diagnostics = (
            NeutralizationDiagnostics(
                n_observations=n_obs,
                n_design_columns=n_columns,
                design_rank=rank,
                condition_number=(
                    condition_number
                ),
                residual_mean_before_winsor=(
                    residual_mean
                ),
                residual_std_before_winsor=(
                    residual_std
                ),
                winsor_lower_bound=float(
                    lower_bound
                ),
                winsor_upper_bound=float(
                    upper_bound
                ),
                n_winsorized_lower=n_lower,
                n_winsorized_upper=n_upper,
                max_abs_final_factor_correlation=(
                    max_abs_correlation
                ),
                final_mean=float(
                    np.mean(
                        alpha_z
                    )
                ),
                final_std=float(
                    np.std(
                        alpha_z,
                        ddof=0,
                    )
                ),
            )
        )

        return NeutralizationResult(
            alpha_z=pd.Series(
                alpha_z,
                index=index,
                name="alpha_z",
                dtype=float,
            ),

            raw_residual=pd.Series(
                raw_residual,
                index=index,
                name="raw_residual",
                dtype=float,
            ),

            winsorized_residual=pd.Series(
                winsorized,
                index=index,
                name="winsorized_residual",
                dtype=float,
            ),

            orthogonalized_residual=pd.Series(
                orthogonalized,
                index=index,
                name="orthogonalized_residual",
                dtype=float,
            ),

            fitted_systematic_score=pd.Series(
                fitted,
                index=index,
                name="fitted_systematic_score",
                dtype=float,
            ),

            coefficients=pd.Series(
                coefficients_array,
                index=design_columns,
                name="coefficient",
                dtype=float,
            ),

            continuous_factor_scaling=(
                scaling
            ),

            sector_baseline=(
                baseline_sector
            ),

            factor_correlations=(
                factor_correlations
            ),

            diagnostics=diagnostics,
        )

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------

    def _validate_and_prepare(
        self,
        cross_section: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            cross_section,
            pd.DataFrame,
        ):
            raise NeutralizationInputError(
                "cross_section must be a pandas DataFrame"
            )

        required_columns = (
            self.config.raw_score_column,
            self.config.size_column,
            self.config.momentum_column,
            self.config.value_column,
            self.config.sector_column,
        )

        missing_columns = [
            column
            for column in required_columns
            if column
            not in cross_section.columns
        ]

        if missing_columns:
            raise NeutralizationInputError(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

        if not cross_section.index.is_unique:
            raise NeutralizationInputError(
                "Security index must be unique"
            )

        if cross_section.index.hasnans:
            raise NeutralizationInputError(
                "Security index contains missing values"
            )

        if (
            len(cross_section)
            < self.config.min_observations
        ):
            raise NeutralizationInputError(
                "Insufficient cross-sectional observations: "
                f"{len(cross_section)} < "
                f"{self.config.min_observations}"
            )

        frame = (
            cross_section
            .loc[
                :,
                list(
                    required_columns
                ),
            ]
            .copy()
        )

        numeric_columns = (
            self.config.raw_score_column,
            self.config.size_column,
            self.config.momentum_column,
            self.config.value_column,
        )

        for column in numeric_columns:
            try:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="raise",
                ).astype(float)

            except Exception as exc:
                raise NeutralizationInputError(
                    f"Column {column!r} must be numeric"
                ) from exc

            values = (
                frame[column]
                .to_numpy(
                    dtype=float
                )
            )

            if not np.all(
                np.isfinite(
                    values
                )
            ):
                raise NeutralizationInputError(
                    f"Column {column!r} contains "
                    "NaN or infinite values"
                )

        if frame[
            self.config.sector_column
        ].isna().any():
            raise NeutralizationInputError(
                "Sector column contains missing values"
            )

        frame[
            self.config.sector_column
        ] = (
            frame[
                self.config.sector_column
            ]
            .astype(str)
            .str.strip()
        )

        if (
            frame[
                self.config.sector_column
            ]
            .eq("")
            .any()
        ):
            raise NeutralizationInputError(
                "Sector values must not be empty"
            )

        if (
            frame[
                self.config.sector_column
            ]
            .nunique()
            < 2
        ):
            raise NeutralizationInputError(
                "At least two sectors are required"
            )

        return frame

    # ----------------------------------------------------------------------
    # Design matrix
    # ----------------------------------------------------------------------

    def _build_design_matrix(
        self,
        frame: pd.DataFrame,
    ) -> Tuple[
        np.ndarray,
        Tuple[str, ...],
        Mapping[
            str,
            FactorStandardization,
        ],
        str,
    ]:

        matrix_columns = [
            np.ones(
                len(frame),
                dtype=float,
            )
        ]

        design_names = [
            "intercept"
        ]

        scaling: Dict[
            str,
            FactorStandardization,
        ] = {}

        continuous_columns = (
            self.config.size_column,
            self.config.momentum_column,
            self.config.value_column,
        )

        for column in continuous_columns:

            values = (
                frame[column]
                .to_numpy(
                    dtype=float
                )
            )

            mean = float(
                np.mean(
                    values
                )
            )

            std = float(
                np.std(
                    values,
                    ddof=0,
                )
            )

            if (
                not math.isfinite(std)
                or std
                <= np.finfo(float).eps
            ):
                raise NeutralizationInputError(
                    f"Risk factor {column!r} "
                    "has zero cross-sectional variance"
                )

            standardized = (
                values
                - mean
            ) / std

            matrix_columns.append(
                standardized
            )

            design_names.append(
                f"z_{column}"
            )

            scaling[column] = (
                FactorStandardization(
                    mean=mean,
                    std=std,
                )
            )

        # Treatment coding:
        #
        # intercept + K-1 sector dummies
        #
        # avoids the dummy-variable trap.
        sectors = tuple(
            sorted(
                frame[
                    self.config.sector_column
                ]
                .unique()
            )
        )

        baseline_sector = (
            sectors[0]
        )

        for sector in sectors[1:]:

            dummy = (
                frame[
                    self.config.sector_column
                ]
                .eq(
                    sector
                )
                .astype(float)
                .to_numpy()
            )

            matrix_columns.append(
                dummy
            )

            design_names.append(
                f"sector[{sector}]"
            )

        design = np.column_stack(
            matrix_columns
        )

        return (
            design,
            tuple(
                design_names
            ),
            scaling,
            baseline_sector,
        )

    # ----------------------------------------------------------------------
    # Neutrality diagnostics
    # ----------------------------------------------------------------------

    def _calculate_factor_correlations(
        self,
        *,
        frame: pd.DataFrame,
        alpha_z: np.ndarray,
    ) -> Mapping[str, float]:

        correlations: Dict[
            str,
            float,
        ] = {}

        continuous_columns = (
            self.config.size_column,
            self.config.momentum_column,
            self.config.value_column,
        )

        for column in continuous_columns:

            factor = (
                frame[column]
                .to_numpy(
                    dtype=float
                )
            )

            correlations[column] = (
                _pearson_correlation(
                    alpha_z,
                    factor,
                )
            )

        sectors = tuple(
            sorted(
                frame[
                    self.config.sector_column
                ]
                .unique()
            )
        )

        # Test ALL sector dummies, including the omitted baseline.
        #
        # Because the residual is orthogonal to the intercept and every
        # included K-1 dummy, it must also be orthogonal to the omitted
        # baseline dummy:
        #
        #     D_baseline = 1 - sum(D_other)
        #
        for sector in sectors:

            dummy = (
                frame[
                    self.config.sector_column
                ]
                .eq(
                    sector
                )
                .astype(float)
                .to_numpy()
            )

            correlations[
                f"sector[{sector}]"
            ] = (
                _pearson_correlation(
                    alpha_z,
                    dummy,
                )
            )

        return correlations


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def neutralize_cross_section(
    cross_section: pd.DataFrame,
    *,
    config: Optional[
        NeutralizerConfig
    ] = None,
) -> NeutralizationResult:
    """
    Functional wrapper around FactorNeutralizer.
    """
    return FactorNeutralizer(
        config=config
    ).neutralize(
        cross_section
    )


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if x.shape != y.shape:
        raise ValueError(
            "Correlation arrays must have identical shape"
        )

    x_centered = (
        x
        - np.mean(
            x
        )
    )

    y_centered = (
        y
        - np.mean(
            y
        )
    )

    denominator = math.sqrt(
        float(
            np.dot(
                x_centered,
                x_centered,
            )
        )
        * float(
            np.dot(
                y_centered,
                y_centered,
            )
        )
    )

    if (
        not math.isfinite(
            denominator
        )
        or denominator
        <= np.finfo(float).eps
    ):
        raise NeutralizationInputError(
            "Cannot calculate Pearson correlation "
            "against a constant factor"
        )

    correlation = float(
        np.dot(
            x_centered,
            y_centered,
        )
        / denominator
    )

    if not math.isfinite(
        correlation
    ):
        raise NeutralizationError(
            "Non-finite correlation produced"
        )

    return correlation
