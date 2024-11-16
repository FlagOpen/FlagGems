import logging


def PASS(e):
    pass


class ErrorHandler:
    def __init__(self, callback_fn=None):
        logging.basicConfig(
            level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        if callback_fn is not None:
            self.__handle_callback(callback_fn)

    def backend_not_support(self, device_name, backend_list):
        raise RuntimeError(f"The {device_name} device is not support currently. ")

    def device_not_found(self):
        raise RuntimeError(
            "No devices were detected on your machine ! \n "
            "Please check that your driver is complete. "
        )

    def register_error(self):
        raise RuntimeError(
            "An error was encountered while registering the triton operator \n "
        )

    def __handle_callback(self, callback_fn):
        pass