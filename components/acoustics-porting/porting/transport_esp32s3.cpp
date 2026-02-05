#include "hal/transport.hpp"

#include "board/board_detector.h"
#include "transport/uart_1.hpp"
#include "transport/uart_console.hpp"

namespace bridge {

void __REGISTER_TRANSPORTS__()
{
    if (DYN_BOARD_TYPE_FROM_I2C_ONCE == porting::BoardType::XIAO_S3)
    {
        [[maybe_unused]] static porting::TransportUART1 transport_uart_1;
    }

    [[maybe_unused]] static porting::TransportUARTConsole transport_uart_console;
}

} // namespace bridge
