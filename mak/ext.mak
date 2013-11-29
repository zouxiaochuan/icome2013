define build_lib_from_dir
	rm -f *.o
	$(CC) $(CFLAGS_COMMON) -c $(2)
	ar rvs $(1) *.o
	rm -f *.o
endef

ext-targets := $(DIR_VLFEAT)/libvlfeat.a \
               $(MODUL_MAXFLOW)

all : $(ext-targets)

vlfeat-srcs = $(wildcard $(DIR_VLFEAT)/vl/*.c)
maxflow-srcs = $(wildcard $(DIR_MAXFLOW)/*.cpp)

$(DIR_VLFEAT)/libvlfeat.a : $(vlfeat-srcs)
	$(call build_lib_from_dir,$@,$^)

$(MODUL_MAXFLOW) : $(maxflow-srcs)
	$(call build_lib_from_dir,$@,$^)

