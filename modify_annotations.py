import os

def modify_annotations(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            modified_lines = []
            for line in lines:
                if line.startswith('3 '):  # Keep only 'Tooth' class
                    # Change class ID from '3' to '0'
                    modified_line = '0' + line[1:]
                    modified_lines.append(modified_line)
            with open(filepath, 'w') as f:
                f.writelines(modified_lines)

# Assuming annotations are in 'labels' directories
# modify_annotations(r'E:\teeth_dataset_afterlook\train\labels')
# modify_annotations('E:\teeth_dataset_afterlook\valid\labels')
modify_annotations(r"E:\teeth_dataset_afterlook\valid\labels")