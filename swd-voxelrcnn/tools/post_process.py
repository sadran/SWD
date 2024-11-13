import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_box_by_position(positions, distance_errors, rotation_errors, save_dir):
    positions_symbols = [chr(i) for i in range(65, 81)] # [A, B, ..., P]
    # Plotting
    plt.figure(figsize=(14, 6))

    # Distance Error Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=positions, y=distance_errors, order=positions_symbols)
    plt.title('Distance Error by Position')
    plt.xlabel('Position')
    plt.ylabel('Distance Error')

    # Rotation Error Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=positions, y=rotation_errors, order=positions_symbols)
    plt.title('Rotation Error by Position')
    plt.xlabel('Position')
    plt.ylabel('Rotation Error')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/box_plot_by_position.png')


def plot_box(distance_errors, rotation_errors, save_dir):
    positions_symbols = [chr(i) for i in range(65, 81)] # [A, B, ..., P]
# Plotting
    plt.figure(figsize=(14, 6))

    # Distance Error Box Plot
    plt.subplot(1, 2, 1)
    plt.boxplot(distance_errors)
    plt.title('Distance Error by Position')
    plt.xlabel('Position')
    plt.ylabel('Distance Error')

    # Rotation Error Box Plot
    plt.subplot(1, 2, 2)
    plt.boxplot(rotation_errors)
    plt.title('Rotation Error by Position')
    plt.xlabel('Position')
    plt.ylabel('Rotation Error')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/box_plot.png')


def find_outliers_and_best_centered_frames_by_position(distance_errors, rotation_errors, positions, indices, save_dir):
    top_selection_number = 5
    
    # Create a DataFrame to hold the errors and positions
    df = pd.DataFrame({
        'index': indices,
        'position': positions,
        'distance_error': distance_errors,
        'rotation_error': rotation_errors
    })

    # Group by position
    grouped = df.groupby('position')
    
    # Initialize lists to store the results for all positions
    all_top_distance_outliers = []
    all_top_rotation_outliers = []
    all_best_distance_frame_indices = []
    all_best_rotation_frame_indices = []

    # Loop over each group (position)
    for pos, group in grouped:
        distance_vals = group['distance_error'].values
        rotation_vals = group['rotation_error'].values

        # Calculate mean and standard deviation
        mean_distance = np.mean(distance_vals)
        std_distance = np.std(distance_vals)

        mean_rotation = np.mean(rotation_vals)
        std_rotation = np.std(rotation_vals)

        # Define outliers (Z-score method)
        z_distance = (distance_vals - mean_distance) / std_distance
        z_rotation = (rotation_vals - mean_rotation) / std_rotation

        # Get the top 5 outliers based on absolute Z-score values
        top_distance_outliers = group.iloc[np.argsort(np.abs(z_distance))[-top_selection_number:]]['index'].tolist()
        top_rotation_outliers = group.iloc[np.argsort(np.abs(z_rotation))[-top_selection_number:]]['index'].tolist()

        all_top_distance_outliers.append(f'Position {pos}: {str(top_distance_outliers)}')
        all_top_rotation_outliers.append(f'Position {pos}: {str(top_rotation_outliers)}')

        # Identify best-centered frames (closest to mean)
        best_distance_indices = np.argsort(np.abs(distance_vals - mean_distance))[:top_selection_number]  # Top 5 closest to mean
        best_rotation_indices = np.argsort(np.abs(rotation_vals - mean_rotation))[:top_selection_number]
        best_distance_frame_indices = group.iloc[best_distance_indices]['index'].tolist()  # Indices
        best_rotation_frame_indices = group.iloc[best_rotation_indices]['index'].tolist()  # Indices

        all_best_distance_frame_indices.append(f'Position {pos}: {str(best_distance_frame_indices)}')
        all_best_rotation_frame_indices.append(f'Position {pos}: {str(best_rotation_frame_indices)}')

    # Write all results to files outside the loop
    with open(f'{save_dir}/top_distance_outliers_by_positions.txt', 'w') as f:
        f.write('\n'.join(all_top_distance_outliers) + '\n')

    with open(f'{save_dir}/top_rotation_outliers_by_positions.txt', 'w') as f:
        f.write('\n'.join(all_top_rotation_outliers) + '\n')

    with open(f'{save_dir}/best_distances_by_positions.txt', 'w') as f:
        f.write('\n'.join(all_best_distance_frame_indices) + '\n')

    with open(f'{save_dir}/best_rotations_by_positions.txt', 'w') as f:
        f.write('\n'.join(all_best_rotation_frame_indices) + '\n')



def find_outliers_and_best_centered_frames(distance_errors, rotation_errors, positions, indices, save_dir):
    top_selection_number = 20
    # Create a DataFrame to hold the errors and indices
    df = pd.DataFrame({
        'index': indices,
        'distance_error': distance_errors,
        'rotation_error': rotation_errors
    })

    # Calculate mean and standard deviation for distance errors
    mean_distance = np.mean(distance_errors)
    std_distance = np.std(distance_errors)

    # Calculate mean and standard deviation for rotation errors
    mean_rotation = np.mean(rotation_errors)
    std_rotation = np.std(rotation_errors)

    # Define outliers (Z-score method)
    z_distance = (distance_errors - mean_distance) / std_distance
    z_rotation = (rotation_errors - mean_rotation) / std_rotation

    # Get the top 5 outliers based on absolute Z-score values
    top_distance_outliers = df.iloc[np.argsort(np.abs(z_distance))[-top_selection_number:]]['index'].tolist()
    top_rotation_outliers = df.iloc[np.argsort(np.abs(z_rotation))[-top_selection_number:]]['index'].tolist()
    with open(f'{save_dir}/top_distance_outliers.txt', 'w') as f:
        f.write(str(top_distance_outliers) + '\n')
    with open(f'{save_dir}/top_rotation_outliers.txt', 'w') as f:
        f.write(str(top_rotation_outliers) + '\n')

    # Identify best-centered frames (closest to mean)
    best_distance_indices = np.argsort(np.abs(distance_errors - mean_distance))[:5]  # Top 5 closest to mean
    best_rotation_indices = np.argsort(np.abs(rotation_errors - mean_rotation))[:5]

    best_distance_frame_indices =  df.iloc[best_distance_indices]['index'].tolist()  # Indices
    best_rotation_frame_indices = df.iloc[best_rotation_indices]['index'].tolist()   # Indices
    
    with open(f'{save_dir}/best_distances.txt', 'w') as f:
        f.write(str(best_distance_frame_indices) + '\n')
    with open(f'{save_dir}/best_rotations.txt', 'w') as f:
        f.write(str(best_rotation_frame_indices) + '\n')


def main():
    result_dir = Path("/Users/sadra/Projects/SW_Detection/SWD/swd-voxelrcnn/output/voxel_rcnn/voxel_rcnn_sw/rotation_along_x_big_dataset_2h/eval/epoch_80/test/validation_set/final_result/data")
    output_dir = result_dir.parent.parent / 'statistics'
    output_dir.mkdir(exist_ok=True)

    ground_truth_path = "/Users/sadra/Projects/SW_Detection/SWD/swd-voxelrcnn/data/swd/validation/ground_truth.csv"
    gt_df = pd.read_csv(ground_truth_path, index_col='frame')

    
    distance_errors, rotation_errors, indices, positions = [], [], [], []

    for pred_file in result_dir.iterdir():
        indx = pred_file.stem
        pos = indx[0]

        # loading ground truth dataframe 
        gt = gt_df.loc[indx]
        gt_center = np.fromstring(gt.sw_translation_vector[1:-1], sep=' ')
        gt_angles = np.radians(np.fromstring(gt.sw_angles[1:-1], sep=' '))
        direction_cosines = np.cos(gt_angles)
        gt_rot = - np.arctan2(direction_cosines[1], direction_cosines[2])

        # load prediction from file
        pred = np.loadtxt(pred_file, dtype='object')
        pred = pred[1:].astype(float)  # Convert the remaining elements to float

        # calculating errors
        distance_error = np.linalg.norm(gt_center - pred[:3])
        rotation_error = (np.abs(gt_rot - pred[6]))

        distance_errors.append(distance_error)
        rotation_errors.append(rotation_error)
        positions.append(pos)
        indices.append(indx)

    print(f"mean_distanc_error: {np.mean(distance_errors)}")
    print(f"mean_rotation_error: {np.mean(rotation_errors)}")

    find_outliers_and_best_centered_frames(distance_errors, rotation_errors, positions, indices, output_dir)
    find_outliers_and_best_centered_frames_by_position(distance_errors, rotation_errors, positions, indices, output_dir)
    plot_box(distance_errors, rotation_errors, output_dir)
    plot_box_by_position(positions, distance_errors, rotation_errors, output_dir)

    


if __name__ == "__main__":
    main()
