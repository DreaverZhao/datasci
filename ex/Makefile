CONVERT = jupyter nbconvert --to html
SRCS1 = $(wildcard ex*.ipynb) climate.ipynb

%.html: %.ipynb
	$(CONVERT) $*

all: $(SRCS1:.ipynb=.html) 

