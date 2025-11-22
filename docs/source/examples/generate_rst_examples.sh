#!/bin/bash
# Run this inside docs/source/examples/
# chmod +x generate_rst_examples.sh
# ./generate_rst_examples.sh

# Loop over all .py files in the current folder
for pyfile in *.py; do
    # Remove .py extension
    base="${pyfile%.py}"
    rstfile="${base}.rst"

    # Create title and underline
    title="${base} Example"
    underline=$(printf '=%.0s' $(seq 1 ${#title}))

    # Write the .rst file
    echo "$title
$underline

.. literalinclude:: ${pyfile}
    :language: python
    :linenos:
" > "$rstfile"

    echo "Generated $rstfile from $pyfile"
done
