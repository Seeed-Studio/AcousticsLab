#include "hal/transport.hpp"

#include "transport/uart_1.hpp"
#include "transport/uart_console.hpp"

namespace bridge {

void __REGISTER_TRANSPORTS__()
{
    [[maybe_unused]] static porting::TransportUARTConsole transport_uart_console;
    [[maybe_unused]] static porting::TransportUART1 transport_uart_1;
}

} // namespace bridge
