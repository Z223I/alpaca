import os
import sys
import argparse
import glob

def isValidDir( parser, argName, directory ):
      """
         Check if valid directory.
      """

      if( False == os.path.exists( directory ) ):
         try:
            os.makedirs( directory, exist_ok=True )
         except OSError as e:
            parser.error( "Can't create directory (--%s=%s)" % ( argName, directory ) )
            print(f'\n\n{e}')

      return os.path.realpath( os.path.abspath( directory ) )

def isValidFile( parser, argName, filename ):
   """
      Check if valid file.
   """

   if( False == os.path.realpath( filename ) ):
      parser.error( "Invalid file name (--%s=%s)" % ( argName, filename ) )

   return os.path.realpath( os.path.abspath( filename ) )

def setupParser():
   """
      Setup the parser.
   """

   epilog = "Alpaca API 1.0 epilog."
   description = "Provides access to Alpaca brokerage accounts."

   class CustomHelpFormatter( argparse.ArgumentDefaultsHelpFormatter,
                              argparse.RawDescriptionHelpFormatter ):
      # Zip
      pass

   parser = argparse.ArgumentParser( add_help = False,
                                      description = description,
                                      epilog = epilog,
                                      formatter_class = CustomHelpFormatter )

   #............................................................................
   #  REQUIRED arguments
   #............................................................................

   #............................................................................
   #  OPTIONAL arguments
   #............................................................................

   optionalGroup = parser.add_argument_group( "Optional" )

   """
   optionalGroup.add_argument( '-k', '--apiKey',
                               help = 'API Key',
                               default = 'demo',
                               required = False )
   """

   # Verbose option
   optionalGroup.add_argument( "-v", "--verbose",
                               action="store_true",
                               help="Display the help message.",
                               default=False,
                               required=False)

   # Add custom help
   optionalGroup.add_argument( "-h", "--help",
                               action="help",
                               default=argparse.SUPPRESS,
                               help="Display the help message.")

   return parser

def displayArgs( argparseNamespace, stream = None ):
   """
      Displays user args.
   """

   if ( None is stream ):
      stream = sys.stdout

   print( "#%s" % ( '=' * 79 ), file = stream )

   maxKeyWidth = 0
   for key in vars( argparseNamespace ):
      if( maxKeyWidth <= len( key ) ):
         maxKeyWidth = len( key )

   for key in sorted( vars( argparseNamespace ) ):
      print( "# %*s: %s" % ( maxKeyWidth, key, eval( "argparseNamespace.%s" %
         ( key ) ) ), file = stream )

   print( "#%s" % ( '=' * 79 ), file = stream )

def validateArgs( userArgs, parser ):
   """
      Validate args.
   """

   pass

def parseArgs( parser, userArgs = None ):
   """
      Parse the args.
   """

   print(f'User arguments: {userArgs}')
   args = parser.parse_args( userArgs )

   if( True == args.verbose ):
      displayArgs( args )

   validateArgs( args, parser )

   return args


