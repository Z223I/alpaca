# Market Sentinel

## High Level Requirements

You are a highly skilled web designer, Python developer, and software architect.
You will be using Python and Flask to be compatible with GoDaddy websites.  I will have the appropriate hosting plan.

## Mid Level Requirements

Create ./public_html/index.html to be the "Market Sentinel" web interface.

Use ONLY standard GoDaddy directories:

- public_html/
- cgi-bin/ (Using atoms/molecules directory structure.)
- logs/
- services/

Create README_market_sentinel.md and keep meaningful notes to yourself about this project.  We will be building this project in steps.

Any Alpaca related code that is an atom is to go into ./cgi-bin/atoms/alpaca_api.  Existing code is in ./atoms/api.

Any Alpaca related code that is a molecule is to go into ./cgi-bin/molecules/alpaca_molecules.  Existing code is in ./code.

Only work in the standard GoDaddy directories.

You may use existing code in other directories as context but leave it unchanged.

### Background Information

#### Review

The Apache 2 web server is running.  The ./public_html files can be reached using http://localhost/market_sentinel/.

See APACHE_SETUP.md and keep it up to date.

### Standards

Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

Keep track of the zoom state of the chart and use it when refreshing due to changes in the indicators.
