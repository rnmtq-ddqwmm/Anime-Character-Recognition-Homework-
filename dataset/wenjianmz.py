import os

# -------- USER SETTINGS --------
folder_path = r"C:\Users\User\Downloads\UI\GA_two_character\dataset\Hatsune Miku"  # folder containing files to rename
new_name = "Hatsune Miku"  # new file base name
keep_extension = True  # keep original extension (True / False)
start_index = 1  # start numbering from this index
# --------------------------------

# Check folder exists
if not os.path.isdir(folder_path):
    raise ValueError("Folder does not exist: " + folder_path)

files = os.listdir(folder_path)
files.sort()  # optional: ensure consistent order

index = start_index

for file in files:
    old_path = os.path.join(folder_path, file)

    # Skip if not a file
    if not os.path.isfile(old_path):
        continue

    # Extract extension
    extension = os.path.splitext(file)[1]

    # Decide final filename
    if keep_extension:
        new_filename = f"{new_name}_{index}{extension}"
    else:
        new_filename = f"{new_name}_{index}"

    new_path = os.path.join(folder_path, new_filename)

    # Rename file
    os.rename(old_path, new_path)
    print(f"Renamed: {file} → {new_filename}")

    index += 1

print("Batch rename complete!")
