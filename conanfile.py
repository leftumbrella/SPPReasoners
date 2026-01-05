from conan import ConanFile
from conan.tools.cmake import cmake_layout


class SPPReasoners(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("opencv/4.12.0")

    def layout(self):
        cmake_layout(self)