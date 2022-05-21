# Flow of codebase:

## Setup.py- 
Add dependencies in the requirement array, so that it can be packaged at the end. It is a static file. It won’t change and only when new requirements come in.

## Requirements.txt- 
We can update it or not, for now, we can but it would be redundant later.

## History.md- 
When a package is created then this stores the version updates.

## Scripts folder:
### Execute.py-
The main file to create cmd line interface for the module. Cli method initializes the entire global variable class which is meant to hold all the global properties and variables across the entire project. Creates 1 obj of class with one variable that holds everything. They are in project/configuration.properties , similar for the simulator. It will then be available to the modules. Click is used to create cli command. This will be extended to more commands such as evaluate etc. for every submodule etc. A command can be added to create a dataset where you can tell what to add with args etc. When you ask for –help then you will see the comments to give that info. In a way, this is the starting point of the project. Other scripts are the ones that can run independently without influencing the project in general. They can be moved to utils or execute.py. 

## Resources folder-
Has config files for project and simulator, but might have model-specific configs. Basically, what the project can be used overall. 

## Output-
Embeddings, birds eye view, checkpoints, any other output. 

## Src/core-
### Model/Simulator-
Common for all the modules we have. 

#### Environment.py- 
Singleton class, one obj of the environment created, every agent will have the same env. Init gets properties etc. The controller is created. Inspired from cordial sync main code. Obj can be fetched from here. It randomizes the agent using teleport. This function has to be modified, the way methods are written is according to the previous version and is simple to understand but can be modified if we want. In the start method, it randomizes agent, inits target obj, we create bird view config and we have agents init view. The rest of the code can be gone to other modules but to make it simple things are added there. The start, stop, reset are also used in gym and are standard commands so it is a common config setup. 
In the model folder, we can keep different modules to modularize, obj detection, RL for obj pickup, moving, etc. 

### Core/services:
Used everywhere in the industry, it has certain functionalities that serve the main module. Things that cannot be added to the module but still want it so it is added to services. 

## Utils-
Loading of property, etc



## Other points to figure out and work with:
Additional structuring of the codebase, pylint, standard way of coding, log handler for modules, exception handling
