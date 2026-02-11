import logging
import os
import uuid


def create_working_dir(out_dir, num_pix_C, num_pix_Z):
    try:
        run_hash = uuid.uuid4().hex
        # Create a run hash and create a directory based on that run hash
        logging.info(f"Executing {run_hash} run")
        directory = os.path.join(out_dir, f"{run_hash}_{num_pix_C}x{num_pix_Z}")
        # Ensure base output directory exists (handles missing parents) and create the run directory
        os.makedirs(directory, exist_ok=True)
        for filename in [
            os.path.join(directory, 'inner.csv'),
            os.path.join(directory, 'outer.csv'),
            os.path.join(directory, 'groundtruth.csv'),
            os.path.join(directory, 'idx.csv')
        ]:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError:
                    pass
        return directory
    except OSError as ose:
        logging.exception(ose)
        return None


def get_dat_files(directory: str):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".dat") and os.path.isfile(os.path.join(directory, f))
    ]