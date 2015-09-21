
import numpy as np
import sys, traceback
from gwl_interface import gwl_interface

def main():

    try :

        status = gwl_interface()

    except KeyboardInterrupt:

        print " Killed by KeyboardInterrupt...exiting "

    except Exception:

        traceback.print_exc(file=sys.stdout)

    sys.exit(0)


if __name__ == "__main__" :

    main()   
