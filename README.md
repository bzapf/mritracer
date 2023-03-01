## Requirements

- FreeSurfer needs to be installed.

- FreeSurfer needs to have been run. The FreeSurfer output folder will be used as, e.g., `export WORKDIR=/home/tux/data/Pat42/`.

- The Python scripts require `numpy` and `nibabel`.

- T1 Maps or T1-weighted images (depending on what is available) have been extracted from Dicom files (Cf. Book Volume 1) and put under `INPUTFOLDER=${WORKDIR}/PREREG` in .mgz format

- git needs to be installed if you want to clone this repository ("download the code")




The following folders/files are needed:
```

${WORKDIR}
├── PREREG
|   ├── 20220230_070707.mgz
|   ├── 20220230_080808.mgz
|   ├── ...
├── mri
│   ├── aseg.mgz
```

## Getting started

```
git clone https://github.com/meg-simula/2023-mri2fem-ii.git
```

To update should some bug fixes be pushed, run
```
$ git pull 
```


## Usage

```
$ cd book2/chapters/tracerconcentration/code/
```

Set path to data for simplicity
```
$ export WORKDIR=/home/tux/data/Pat42/
```

### Registration pipeline

In terminal, run

```
bash register.sh
```



### Normalization

We next normalize the (registered) T1 weighted images (divide by median signal in a reference ROI to account for scanner variability over time).
See [2] (Supplementary Material) for details.

```
python normalize.py --inputfolder ${WORKDIR}/REGISTERED/ \
--exportfolder ${WORKDIR}/NORMALIZED_CONFORM/ \
--refroi ${WORKDIR}/mri/refroi.mgz
```

### Create a synthetic T1 map

Create an image with voxel value 1 s everywhere inside the parenchyma:

```
python make_brainmask.py --aseg ${WORKDIR}/mri/aseg.mgz --t1 1
```


### Tracer estimation

To estimate tracer concentrations as in [1] (Supplementary Material),

```
python estimatec.py --inputfolder ${WORKDIR}/NORMALIZED_CONFORM/ \
--exportfolder ${WORKDIR}/CONCS_constT1/ --t1map ${WORKDIR}/mri/synthetic_T1_map.mgz
```
will create a binary mask for the parenchyma under  `${WORKDIR}/mri/parenchyma_only.mgz`.


We can also map out everything outside the brain to get a cleaner image:
```
python estimatec.py --inputfolder ${WORKDIR}/NORMALIZED_CONFORM/ \
--exportfolder ${WORKDIR}/CONCS_T1MAP_MASKED/ --t1map ${WORKDIR}/mri/synthetic_T1_map.mgz \
--mask ${WORKDIR}/mri/parenchyma_only.mgz
```



If a T1 map is available (same resolution as T1 weighted images and registered to T1 weighted images)
```
python estimatec.py --inputfolder ${WORKDIR}/NORMALIZED_CONFORM/ \
--exportfolder ${WORKDIR}/CONCS_T1MAP/ --t1map ${WORKDIR}/<path_to_your_T1_Map> \
--mask ${WORKDIR}/mri/parenchyma_only.mgz
```

## References

[1] Valnes, Lars Magnus et al. "Supplementary Information for "Apparent diffusion coefficient estimates based on 24 hours tracer movement support glymphatic transport in human cerebral cortex", Scientific Reports (2020)

[2] PK Eide et al. "Sleep deprivation impairs molecular clearance from the human brain", Brain (2021)
