from havok import (
    build_havok_model,
    compute_forcing_threshold,
    load_csv_data,
    plot_attractor_vertical_with_activation_slices,
    plot_matrix_and_vector_with_shared_colorbar,
    plot_timeseries_HAVOK_with_slices,
)


def main():
    data = load_csv_data(["../dataset/lorenz/lorenz_x_emb.csv"])[0]
    if data.ndim == 1:
        data = data[None, :]

    dt, delays, rank, num_chaos = 0.001, 100, 15, 1
    A, B, V_test, yout, u_test, t_test = build_havok_model(
        data, dt, delays, rank, num_chaos
    )

    thresh = compute_forcing_threshold(u_test, 95)
    plot_matrix_and_vector_with_shared_colorbar(A, B)
    plot_timeseries_HAVOK_with_slices(
        "Lorenz",
        V_test,
        yout,
        u_test,
        t_test,
        r=rank,
        threshold=thresh,
        min_jump_dist=int(0.5 / dt),
        segment_width=20,
    )
    plot_attractor_vertical_with_activation_slices(
        V_test,
        yout,
        u_test,
        t_test,
        threshold=thresh,
        min_jump_dist=20,
        label="Lorenz",
    )


if __name__ == "__main__":
    main()
