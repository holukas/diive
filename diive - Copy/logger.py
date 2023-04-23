"""

    https://stackoverflow.com/questions/28655198/best-way-to-display-logs-in-pyqt

"""

import logging

# from PyQt5 import QtWidgets as qw


class QTextEditLogger(logging.Handler):
    def __init__(self, parent, lyt):
        super().__init__()
        self.widget = qw.QTextEdit(parent)
        # self.widget = qw.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        lyt.addWidget(self.widget)

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)
        # self.widget.appendPlainText(msg)

def log(name, dict, highlight):
    """ Output to GUI text field (LogBox)

    Parameters
    ----------
    highlight
    """
    whitespace = '&nbsp;' * 8

    if name != '':
        if highlight:
            logging.info(f'<span style="background-color:#FFF9C4;"><b>{name}</b></span>')
        else:
            logging.info(f'<b>{name}</b>')

    for key, value in dict.items():
        logging.info(f'{whitespace}<font color="#0288D1">{key}</font>: {value}')

# logging.debug('damn, a bug')
# logging.info('something to remember')
# logging.warning('that\'s not right')
# logging.error('foobar')

# class Logger(object):
#     def __init__(self, run_id):
#         super(Logger, self).__init__()
#
#         # create logger
#         logfile = '{}.log'.format(run_id)
#         logger = logging.getLogger(logfile)
#         logger.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(message)s')  # create formatter for handlers
#         # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         # create file handler
#         fh = logging.FileHandler(logfile, mode='w')  # create file handler
#         fh.setLevel(logging.INFO)  # logs info messages and above
#         fh.setFormatter(formatter)  # add formatter to the handler
#         logger.addHandler(fh)  # add the handler to logger
#         self.logger = logger
#
#     def log_info(self, record):
#         # outputs to console and log file
#         self.logger.info(record)
#         print(record)
#
#         return None


