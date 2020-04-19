import MySQLdb
from MySQLdb import cursors
import sys

__author__ = 'DColton'


class DBConn:
    """ Some methods to allow easy:
         - Database connection
         - Running select queries
    """

    # Connection variables
    host = 'localhost'
    user = 'root'
    passwd = 'mysqlroot'
    db = 'thesis'

    # Initialise self
    def __init__(self):
        try:
            self.conn = MySQLdb.connect(self.host,
                                        self.user,
                                        self.passwd,
                                        self.db,
                                        use_unicode=True,
                                        charset="utf8",
                                        cursorclass=MySQLdb.cursors.SSDictCursor)

        # Handle database errors
        except MySQLdb.Error, e:
            print "Error %d: %s" % (e.args[0], e.args[1])
            sys.exit(1)

    def close_conn(self):
        self.conn.close()

    def create_conn(self):
        return self.conn
