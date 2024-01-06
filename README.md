# kyoani
analysis code for behavior data 
<br>it looks over each animal for every session, stores session stats in the animal objects, and plot for each animal across sessions (in plots) or for all animals (in behavior). Processing all sessions can take very long so the processed data are binarized in pickle format. One should be able to load as it is in main. It might take some path-editing for the figures folder. It outputs individual animal figures to its own folder and collective figures to a folder outside, with animals divided into groups by different criteria. 
