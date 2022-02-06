
# list all md files from parts folder
files=$(find ./parts -name "*.md" -type f)

# sort files in files variable
files=$(echo $files | tr ' ' '\n' | sort -V)

echo "Generating PDF from $files"

pandoc -o ML_Summary.pdf ML_Summary.md $files
