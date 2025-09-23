import numpy as np

from .build_havok import (
    build_havok_model,
    compute_forcing_threshold,
    compute_svd,
    differentiate,
    load_csv_data,
)

from .havok_lstm import havok_lstm_pipeline_pytorch, havok_lstm_one_step_pipeline

from .plotting import (
    plot_attractor_vertical_with_activation_slices,
    plot_havok_reconstruction,
    plot_matrix_and_vector_with_shared_colorbar,
    plot_timeseries_HAVOK_with_slices,
)

from .havok_dae import (
    DAE_HAVOK,
    make_dataloader,
    train_dae_havok,
    encode_csv_with_dae,
    extract_K,
    compute_residual,
    build_residual_havok,
    hierarchical_predict,
    save_havok_npz,
    save_koopman_npz,
    load_havok_npz,
    load_koopman_npz,
)

from .havok_metrics import spectral_radius, antisymmetry_report, koopman_residual_stats

from havok.havok_kAE import (
    DAE_HAVOK,
    make_dataloader,
    train_dae_havok,
    encode_array_with_dae,
    extract_K,
    build_latent_havok,
    save_koopman_npz,
    save_havok_npz,
)

from havok.havok_KAE_metrics import antisymmetry_report, spectral_radius

from havok.havok_HKAE import (
    HKAEConfig,
    HKAETrainer,
    HavokConfig,
    HavokLatentAnalyzer,
    delay_stack,
)

from havok.post_latent import (
    BackProjectConfig,
    predict_observation_from_latent,
    metrics_latent_havok,
    discrete_to_continuous,
    compare_timeseries_and_plot,
    show_learning_metrics,
    compare_timeseries_and_plot_segmented,
)
