```bash
# 可选
conan profile detect

conan install . -pr=relwithdebinfo --build=missing
cmake --preset conan-release
cmake --build --preset conan-release
```