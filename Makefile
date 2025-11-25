# Clean all generated directories
CLEAN_DIRS = __epyccel__ __pycache__ figs

# Directories to clean
ROOT_DIRS = $(HOME)/pyrefiga/r_adaptive_refinement \
            $(HOME)/pyrefiga/cad\
            $(HOME)/pyrefiga/examples \
            $(HOME)/pyrefiga/docs/source/examples \
            $(HOME)/pyrefiga/docs/source/r_adaptive_refinement \
            $(HOME)/pyrefiga/newFolder \
            $(HOME)/pyrefiga/pyrefiga/tests

# Clean target
clean:
	@for root in $(ROOT_DIRS); do \
		echo "Cleaning in $$root..."; \
		find $$root -type d \( $(foreach dir, $(CLEAN_DIRS), -name "$(dir)" -o) -false \) -exec echo "Removing: {}" \; -exec rm -rf {} +; \
	done
	@echo "Cleanup completed in all specified directories."

