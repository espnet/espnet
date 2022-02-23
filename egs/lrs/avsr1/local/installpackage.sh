#! /usr/bin/env bash 

#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

. ./cmd.sh
. ./path.sh

# hand over parameters 
OPENFACE_DIR=$1			# Path to OpenFace build directory
VIDAUG_DIR=$2 			# Path to vidaug directory
DEEPXI_DIR=$3			# DeepXi directory

conda install -n espnet_venv tensorflow tqdm pysoundfile
conda install -c esri tensorflow-addons
while true
do
    read -r -p "For this recipe you will need sudo right to install OpenFace, do you want to continue? [Y/n] " input
    case $input in 
	[yY][eE][sS]|[yY])
	    mkdir -p local/installations
    	    if [ -d "$OPENFACE_DIR" ] ; then
		echo "OpenFace already installed."
    	    else
    		while true
    		do
        	    read -r -p "Have you already installed OpenFace on your computer [Y/n] " input
        	    case $input in 
	    		[yY][eE][sS]|[yY])
			    echo "Please path OpenFace directory"
			    exit 1;
	        	    ;;
            		[nN][oO]|[nN])
		    	   cd local/installations
		    	   $MAIN_ROOT/tools/installers/install_openface.sh || exit 1;
		    	   cd ../..
			   ;;
        	    esac      
    		done
    	    fi
    	    if [ -d "$VIDAUG_DIR" ] ; then
		echo "Vidaug already installed."
    	    else
    		while true
    		do
        	    read -r -p "Have you already installed Vidaug on your computer [Y/n] " input
        	    case $input in 
	    		[yY][eE][sS]|[yY])
			    echo "Please path Vidaug directory"
			    exit 1;
	        	    ;;
            		[nN][oO]|[nN])
		    	    cd local/installations
		            $MAIN_ROOT/tools/installers/install_vidaug.sh || exit 1;
		    	    cd ../..
			    ;;
        	    esac      
    		done
    	    fi
    	    if [ -d "$DEEPXI_DIR" ] ; then
		echo "DeepXi already installed."
    	    else
    		while true
    		do
        	    read -r -p "Have you already installed DeepXi on your computer [Y/n] " input
         	    case $input in 
	    		[yY][eE][sS]|[yY])
			    echo "Please path DeepXi directory"
			    exit 1;
	        	    ;;
            		[nN][oO]|[nN])
		            cd local/installations
		    	    $MAIN_ROOT/tools/installers/install_deepxi.sh || exit 1;
		    	    cd ../..
			    ;;
        	    esac      
    		done
    	    fi
	    break
	    ;;
        [nN][oO]|[nN])
            exit 1;
	    ;;
        *)
            echo "Invalid choice..."
            exit 1;
            ;;
    esac      
done

exit 0
