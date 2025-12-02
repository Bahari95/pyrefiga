# Clean all generated directories
CLEAN_DIRS = __epyccel__ __pycache__ figs

# Directories to clean
ROOT_DIRS = ./pyrefiga/r_adaptive_refinement \
            ./pyrefiga/cad\
            ./pyrefiga/examples \
            ./pyrefiga/docs/examples \
            ./pyrefiga/newFolder \
            ./pyrefiga/pyrefiga/tests

# Clean target
clean:
	@for root in $(ROOT_DIRS); do \
		echo "Cleaning in $$root..."; \
		find $$root -type d \( $(foreach dir, $(CLEAN_DIRS), -name "$(dir)" -o) -false \) -exec echo "Removing: {}" \; -exec rm -rf {} +; \
	done
	@echo "Cleanup completed in all specified directories."

