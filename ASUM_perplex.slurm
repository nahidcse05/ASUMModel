#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=51200MB
#SBATCH --time=06:20:00
#SBATCH -p economy
#SBATCH --output=/nv/blue/mr4xb/public_html/ASUM/output/consoleLog.txt
#SBATCH --error=/nv/blue/mr4xb/public_html/ASUM/output/consoleEroor.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr4xb@virginia.edu

cd /nv/blue/mr4xb/public_html/ASUM/
module load java/1.8.0_45
java -jar ./ASUM_camera_0.jar
java -jar ./ASUM_camera_1000.jar
java -jar ./ASUM_camera_2000.jar
java -jar ./ASUM_camera_3000.jar
java -jar ./ASUM_camera_4000.jar
java -jar ./ASUM_camera_5000.jar
java -jar ./ASUM_phone_0.jar
java -jar ./ASUM_phone_1000.jar
java -jar ./ASUM_phone_2000.jar
java -jar ./ASUM_phone_3000.jar
java -jar ./ASUM_phone_4000.jar
java -jar ./ASUM_phone_5000.jar
java -jar ./ASUM_tv_0.jar
java -jar ./ASUM_tv_1000.jar
java -jar ./ASUM_tv_2000.jar
java -jar ./ASUM_tv_3000.jar
java -jar ./ASUM_tv_4000.jar
java -jar ./ASUM_tv_5000.jar
java -jar ./ASUM_tablet_0.jar
java -jar ./ASUM_tablet_1000.jar
java -jar ./ASUM_tablet_2000.jar
java -jar ./ASUM_tablet_3000.jar
java -jar ./ASUM_tablet_4000.jar
java -jar ./ASUM_tablet_5000.jar
module unload java/1.8.0_45