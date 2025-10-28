# Build Fix Applied

## Issue
OpenMP build was failing with "undefined reference to `main`" because object files weren't being compiled.

## Root Cause
The C++ compilation rule in the Makefile didn't have OpenMP flags set, so the `.cpp` files weren't being compiled into `.o` object files before linking.

## Fix Applied
Updated Makefile line 194 to include OpenMP flags in the C++ compilation rule:
```makefile
$(CXX) $(CXXFLAGS_BASE) $(OPT_FLAGS) $(OPENMP_FLAGS) $(INCLUDES) -c $< -o $@
```

## Verification
After this fix, the build should work:
```bash
make clean
make openmp
```

## Status
✅ Fixed and ready to use on cluster