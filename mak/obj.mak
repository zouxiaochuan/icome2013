

$(DIR_OBJ)/%.o: $(DIR_SRC)/%.cpp $(DIR_OBJ)/%.d
	$(CXX) $(CFLAGS_COMMON) -c $(<) -o $(@)

$(DIR_OBJ)/%.d: $(DIR_SRC)/%.cpp
	$(CXX) $(CFLAGS_COMMON) -MM -MT '$(DIR_OBJ)/$*.o $(DIR_OBJ)/$*.d' $(<) -MF $(@)


all-srcs := $(wildcard $(DIR_SRC)/*.cpp)
all-objs := $(addprefix $(DIR_OBJ)/, $(notdir $(all-srcs:.cpp=.o)))
deps := $(all-objs:.o=.d)

