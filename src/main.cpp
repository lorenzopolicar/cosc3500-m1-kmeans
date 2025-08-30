#include <iostream>

int main(int argc, char* argv[]) {
    (void)argc; // Suppress unused parameter warning
    std::cout << argv[0] << " stub build ok" << std::endl;
    
    // Print compiler and version info
    std::cout << "Compiler: " << __VERSION__ << std::endl;
    
    return 0;
}
