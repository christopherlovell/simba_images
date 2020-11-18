#!/bin/bash 

#Powderday cluster setup convenience script for SLURM queue mananger
#on COSMA at the Durham.  This sets up the model
#files for a cosmological simulation where we want to model many
#galaxies at once.

#Notes of interest:

#1. This does *not* set up the parameters_master.py file: it is
#assumed that you will *very carefully* set this up yourself.

#2. This requires bash versions >= 3.0.  To check, type at the shell
#prompt: 

#> echo $BASH_VERSION

n_nodes=$1
model_dir=$2
hydro_dir=$3
model_run_name=$4
COSMOFLAG=$5
model_dir_remote=$6
hydro_dir_remote=$7
xpos=$8
ypos=$9
zpos=${10}
galaxy=${11}
snap=${12}
tcmb=${13}
index=${14}
job_flag=${15}
N=${16}
halo=${17}
processornumber=${18}
nprocs=${19}

echo "processing model file for galaxy,snapshot,processor:  $galaxy,$snap,$processornumber"


#clear the pyc files
rm -f *.pyc

#set up the model_**.py file
#echo "setting up the output directory in case it doesnt already exist"
# echo "snap is: $snap"
# echo "model dir is: $model_dir"
if [ ! -d "$model_dir" ]; then
    mkdir $model_dir
    out_folder="$model_dir/out"
    err_folder="$model_dir/err"
    mkdir $out_folder
    mkdir $err_folder
fi

# echo "model output dir is: $model_dir_remote"
if [ ! -d "$model_dir_remote" ]; then
    mkdir $model_dir_remote
fi

# copy over parameters_master file
# cp parameters_master.py $model_dir/.

filem="$model_dir/snap${snap}_${galaxy}_${processornumber}.py"
# echo "writing to $filem"
rm -f $filem

if [ $N -gt 1000 ]; then
    Nout=1000
else
    Nout=${N}
fi

echo "#Snapshot Parameters" >> $filem
echo "#<Parameter File Auto-Generated by setup_all_cluster.sh>" >> $filem
echo "snapshot_num = $snap" >> $filem
echo "galaxy_num = $halo" >>$filem
echo -e "\n" >> $filem


echo "galaxy_num_str = '{:05.0f}'.format(galaxy_num)">>$filem
echo "snapnum_str = '{:03.0f}'.format(snapshot_num)">>$filem
echo -e "\n" >>$filem

if [ $COSMOFLAG -eq 1 ]
then
    echo "hydro_dir = '$hydro_dir_remote/'">>$filem
    echo "snapshot_name = 'subset_'+galaxy_num_str+'.h5'" >>$filem
else
    echo "hydro_dir = '$hydro_dir_remote/'">>$filem
    echo "snapshot_name = 'subset_'+galaxy_num_str+'.h5'" >>$filem
fi


echo -e "\n" >>$filem

echo "#where the files should go" >>$filem
echo "PD_output_dir = '${model_dir_remote}/' ">>$filem
echo "Auto_TF_file = 'snap'+snapnum_str+'.logical' ">>$filem
echo "Auto_dustdens_file = 'snap'+snapnum_str+'.dustdens' ">>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#FILE I/O" >>$filem
echo "#===============================================" >>$filem
echo "inputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'${processornumber}.rtin'" >>$filem
echo "outputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'${processornumber}.rtout'" >>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#GRID POSITIONS" >>$filem
echo "#===============================================" >>$filem
echo "x_cent = ${xpos}" >>$filem
echo "y_cent = ${ypos}" >>$filem
echo "z_cent = ${zpos}" >>$filem

echo -e "\n\n" >>$filem
echo "#===============================================" >>$filem
echo "#CMB INFORMATION" >>$filem
echo "#===============================================" >>$filem
echo "TCMB = ${tcmb}" >>$filem

if [ "$job_flag" -eq 1 ]; then
    echo "writing slurm submission master script file"
    qsubfile="$model_dir/master.snap${snap}_${galaxy}.job"
    rm -f $qsubfile
    echo $qsubfile
    
    echo "#! /bin/bash" >>$qsubfile
    echo "#SBATCH --account narayanan" >>$qsubfile
    echo "#SBATCH --qos narayanan-b" >>$qsubfile
    echo "#SBATCH --job-name=${model_run_name}.snap${snap}.g${galaxy}" >>$qsubfile
    echo "#SBATCH --output=out/pd.master.snap${snap}.g${galaxy}.%a.%N.%j" >>$qsubfile
    echo "#SBATCH --error=err/pd.master.snap${snap}.g${galaxy}.%a.%N.%j" >>$qsubfile
    echo "#SBATCH --mail-type=ALL" >>$qsubfile
    echo "#SBATCH --mail-user=c.lovell@herts.ac.uk" >>$qsubfile
    echo "#SBATCH --time=0-04:00" >>$qsubfile
    echo "#SBATCH --nodes=1">>$qsubfile
    echo "#SBATCH --ntasks=32">>$qsubfile
    echo "#SBATCH --mem-per-cpu=3800">>$qsubfile
    echo "#SBATCH --array=0-${nprocs}">>$qsubfile
    echo -e "\n">>$qsubfile
    echo -e "\n" >>$qsubfile
    
    echo "module purge">>$qsubfile
    echo "module load git/2.14.1 gcc/8.2.0 openmpi/4.0.1 hdf5/1.10.1">>$qsubfile
    echo "module load conda/4.8.3">>$qsubfile
    echo -e "\n">>$qsubfile
    
    echo "source ~/.bashrc">>$qsubfile
    echo "source activate pday">>$qsubfile
    echo -e "\n">>$qsubfile
    
    echo "PD_FRONT_END=\"/home/c.lovell/codes/powderday/pd_front_end.py\"">>$qsubfile
    echo -e "\n">>$qsubfile
    
    echo "python \$PD_FRONT_END . parameters_master snap${snap}_${galaxy}_\$SLURM_ARRAY_TASK_ID > gal_${galaxy}/snap${snap}_${galaxy}_\$SLURM_ARRAY_TASK_ID.LOG">>$qsubfile
    echo -e "\n">>$qsubfile
    
    echo "echo \"Job done, info follows...\"">>$qsubfile
    echo "sacct -j \$SLURM_JOBID --format=JobID,JobName,Partition,Elapsed,ExitCode,MaxRSS,CPUTime,SystemCPU,ReqMem">>$qsubfile
    echo "exit">>$qsubfile
fi

#done