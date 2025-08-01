# Future Bracket Order

## High Level Requirements

You are a highly skilled PRD author, Python developer, and software architect.
You are only to create the PRD specs/use_config_py_vs_env_variables_prd.md

## Mid Level Requirements

### Document in the PRD

#### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

#### Task

The current code/alpaca.py code uses environment variables. Update the code to a config file ./code/.alpaca_config.py that contains a dataclass.


## Low Level Requirements

### Document in the PRD

Add ./code/.alpaca_config.py to .gitignore

Use this JSON for creating the data class in ./code/.alpaca_config.py

{
  "Provider": {
    "<provider_name>": {
      "<account_name>": {
        "paper": {
          "APP Key": "<APP Key>",
          "APP Secret": "<APP Secret>"
          "URL": "<URL>"
        },
        "live": {
          "APP Key": "<APP Key>",
          "APP Secret": "<APP Secret>"
          "URL": "<URL>"
        },
        "cash": {
          "APP Key": "<APP Key>",
          "APP Secret": "<APP Secret>"
          "URL": "<URL>"
        }
      }
    }
  }
}