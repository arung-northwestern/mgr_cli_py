## cli overview
cli should have name mcli, three core commands, 
- process :: to process one or more input structures 
- config-building start the interactive config builder that writes an correctly formatted input file that can be reusable or for process mulitple materials
- generate-config : to generate a default config, but with optionally changing some configs, like mcli generate-config -mt 1.5 to use a differnet mask thickness while generating config.    

## core working backend
- lib.rs : do_periodic_segmenation_rust function and its helper functions all in lib.rs and working.

##  input params 
- spacing  grid spacing in angstroms default : 0.5
- h : minimum difference between pokcet size and smallest window radii in angstroms default 0.2
- mask-thickness : mask thickness for watershed regions segmentation. anythin closer than this to framework surface becomes part of the framework. farther becomes the void space. Default value 0.5. 
- Compactness. Do not expose this parameter in the CLI. Defaults to zero. 
- Wall thickness This is the percentage of the perpendicular lengths of the unit cell. That needs to be considered as the war region. Defaults to 0.08. Or 8%. 
- return dgrid Whether to return. Distance grid. As one of the outputs. In the output dictionary. 
- Returned padded regions. Whether to return the padded regions segmentation grid as one of the outputs. Defaults to false. 
- Computer connections whether to compute connections between regions or not. Defaults to true. 
- threshold_abs: The minimum size in Angstrom needed for a maxima to be Detected. Defaults to 1.5 angstroms. 
- min_distance: Minimum distance of separation needed between maxima for them to not get merged. Defaults to. 1.5 Angstroms. 
- minimum_window_separation: Minimum separation between windows for them to not get merged, in angstroms, Defaults to. 1.0 Angstroms. 

Each of these options need to have shorthand as well, for example -mt for mask_thickness, -ta for threshold absolute -ms for minimum separation etc. -spc for spacing etc.

-i for path to input cif file, -o for output folder path, if no -o is specified output to current dir. 

cli tool should work with one cif or multiple cifs as inputs. if multiple cifs are passed, parallelize over the inputs. 

## outputs
- make npz compressed python dictionary that can be read in. Dictionary has same keys as the do periodic segemntation rust function from src/lib.rs

## test CIFS 
- use ZIF-8-RASPA.cif uder cifs/ folder for single material testing. for multiple material testing using ZIF-8-RASPA.cif, P1-Cu-BTC.cif, and LTA.cif in tests folder

## general rules
- do not edit lib.rs file directly. 
- do not make unnecessary test files, ask if you have to. 
- use code-index mcp to find functions quickly instead of broad grep
- use cipher mcp to store key information under the tag 'mgr_cli_py'
- do not edit the cli_needs.md file
- ask if unsure about any arguments or funtionality in the rust source or for the cli tool