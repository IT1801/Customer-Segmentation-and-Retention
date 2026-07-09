import sys

class CSRException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)

        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return (
            f"Error occurred in Python script [{self.file_name}] "
            f"at line [{self.lineno}] "
            f"with message [{self.error_message}]"
        )