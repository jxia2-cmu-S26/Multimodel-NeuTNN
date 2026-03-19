from pathlib import Path
import sys


def main():
    """
    Batch over all ERP IMU files under sourcedata and generate head IMU figures.

    Rules:
    - Only process files whose names contain 'task-ERP_imu';
    - Prefer using .vhdr as the entry point (which resolves the corresponding .dat);
    - For each file, generate one figure saved next to the IMU files as:
      sub-XX_ses-YY_task-ERP_imu_head.png
    """
    # Root path of sourcedata under the project
    sourcedata_root = Path(
        r"D:\xia\CMU\18743-Neuromorphic\Multi_Sensory_NeuTNN\osfstorage-archive\sourcedata"
    )

    from read_imu_dat import (
        load_imu_brainvision,
        plot_imu_head,
        plot_imu_raw_vs_denoised,
    )

    # Find all ERP IMU .vhdr files
    vhdr_files = sorted(
        sourcedata_root.rglob("*task-ERP_imu.vhdr")
    )

    print(f"Found {len(vhdr_files)} ERP IMU header files; generating figures one by one.")

    for vhdr_path in vhdr_files:
        try:
            print("\n=== Processing file ===")
            print("VHDR:", vhdr_path)
            data, info = load_imu_brainvision(vhdr_path)

            # Save head IMU figure next to the .vhdr/.dat file
            save_path = vhdr_path.with_name(vhdr_path.stem + "_head.png")
            print("Saving head figure to:", save_path)

            # duration_sec=None => use full duration
            plot_imu_head(
                data,
                info,
                duration_sec=None,
                save_path=save_path,
                show=False,
            )

            # Also generate raw vs denoised figure (duration fixed to 10 s)
            save_rvd = vhdr_path.with_name(vhdr_path.stem + "_raw_vs_denoised_H_X_10s.png")
            print("Saving raw-vs-denoised figure to:", save_rvd)
            plot_imu_raw_vs_denoised(
                data_raw=data,
                info=info,
                data_denoised=None,
                window_length=11,
                polyorder=3,
                duration_sec=10.0,
                save_path=save_rvd,
                place="H",
                axis="X",
                show=False,
            )
        except Exception as exc:
            print(f"Error while processing {vhdr_path}: {exc}")


if __name__ == "__main__":
    main()

