# Define colors
GREEN=\033[0;32m
NC=\033[0m

# List of datasets
DATASETS=apps humaneval

all: analysis figures

analysis:
	@for DS in $(DATASETS); do \
		printf "Running $(GREEN)$$DS$(NC) analysis script.\n"; \
		cd data/$$DS && bash run-analysis.sh && cd -; \
	done
	@printf "${GREEN}Done.${NC}"

figures:
	@for DS in $(DATASETS); do \
		printf "Making ${GREEN}$$DS${NC} figures.\n"; \
		cd data/$$DS && bash make-figs.sh && cd -; \
	done
	@printf "${GREEN}Done.${NC}"

clean:
	@rm -rfv ./figures
	@for DS in $(DATASETS); do \
		rm -rfv  ./data/$$DS/processed; \
	done

.PHONY: all analysis figures clean
