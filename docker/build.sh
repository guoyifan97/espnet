#!/usr/bin/env bash

# 2019, Nelson Yalta
# 2019, Ludwig Kürzinger, Technische Universität München


SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Default values
ubuntu_ver=20.04
cuda_ver=11.1
build_ver=cpu
build_cores=24
th_ver=1.8.0

docker_ver=$(docker version -f '{{.Server.Version}}')
echo "Using Docker Ver.${docker_ver}"


cmd_usage() {
    PROGRAM=$(basename "$0")
    cat >&2 <<-_EOF
    DESCRIPTION
        A script for automatic builds of docker images for ESPnet and
        pushing them to Dockerhub.
        Also able to build containers based on local build configuration.
    
    USAGE
        ${PROGRAM} <args> <mode>
        ${PROGRAM} build_and_push
        ${PROGRAM} --build-ver [cpu|gpu] local

            mode      Select script functionality
            args      Set up building features

        Modes
            build           build docker containers
            test            test docker containers
            push            push to docker hub
            build_and_push  automated build, test and push
            local           build a docker container from the local ESPnet repository
                            using the base image from Docker Hub (espnet/espnet:runtime)
                            optional: cpu or CUDA version (default: cpu)
            fully_local     like local, but also builds the base image
        
        Arguments
            build-ver       cpu/gpu
            ubuntu-ver      any ubuntu version available at docker hub (e.g. 18.04/20.04/...)
                            (default: 18.04)
            cuda-ver        any cuda version available at nvidia (e.g. 9.0/9.1/...)
                            (default: 10.1)
            build-cores     cores employed for building the container
            th-ver          Pytorch version for fully local build

    CAVEATS
        For local builds, the image pulled from Docker Hub is based on Ubuntu 16,
        whereas a fully_local build uses the distribution specified in
        prebuilt/runtime/Dockerfile (currently set to Ubuntu 18.04).

	_EOF
    exit 1
}


build(){
    echo "Build Latest docker containers"
    # build runtime and gpu based containers
    this_tag=espnet/espnet:runtime-latest
    docker_image=$( docker images -q  ${this_tag} )
    if ! [[ -n ${docker_image} ]]; then
        echo "Now building Runtime container"
        docker build --build-arg DOCKER_VER=${docker_ver} \
                    --build-arg FROM_TAG=${default_ubuntu_ver} \
                    --build-arg NUM_BUILD_CORES=${build_cores} \
                    -f prebuilt/runtime/Dockerfile -t ${this_tag} . | tee -a build_runtime.log > /dev/null

        docker_image=$( docker images -q ${this_tag} )
        [ -z "${docker_image}" ] && exit 1
    fi

    this_tag=espnet/espnet:cuda-latest
    docker_image=$( docker images -q  ${this_tag} )
    if ! [[ -n ${docker_image} ]]; then
        echo "Now building CUDA container"
        docker build --build-arg FROM_TAG=runtime-latest \
                    -f prebuilt/devel/gpu/${default_cuda_ver}/Dockerfile -t ${this_tag} . | tee -a build_cuda.log > /dev/null
        docker_image=$( docker images -q ${this_tag} )
        [ -z "${docker_image}" ] && exit 1
    fi

    # build cpu based
    docker_image=$( docker images -q espnet/espnet:cpu-latest )
    this_tag=espnet/espnet:cpu-latest
    docker_image=$( docker images -q  ${this_tag} )
    if ! [[ -n ${docker_image} ]]; then
        echo "Now building cpu-latest with ubuntu:${default_ubuntu_ver}"
        docker build --build-arg FROM_TAG=runtime-latest -f prebuilt/devel/Dockerfile -t ${this_tag} . | tee -a build_cpu.log > /dev/null

        docker_image=$( docker images -q ${this_tag} )
        [ -z "${docker_image}" ] && exit 1
    fi

    # build gpu based
    build_args="--build-arg FROM_TAG=cuda-latest 
                --build-arg CUDA_VER=${default_cuda_ver}"
    this_tag=espnet/espnet:gpu-latest
    docker_image=$( docker images -q ${this_tag}  )
    if ! [[ -n ${docker_image} ]]; then
        echo "Now building gpu-latest with ubuntu:${default_ubuntu_ver} and cuda:${default_cuda_ver}"
        docker build ${build_args} -f prebuilt/devel/Dockerfile -t ${this_tag}  . | tee -a build_gpu.log > /dev/null
        docker_image=$( docker images -q ${this_tag} )
        [ -z "${docker_image}" ] && exit 1
    fi
}


build_local(){
    echo "Building docker container: base image, and image for ${build_ver}"
    sleep 1

    # prepare espnet-repo, assuming that this script is in folder espnet/docker
    cd ${SCRIPTPATH}/..
    ESPNET_ARCHIVE="./espnet-local.tar"
    echo "Reconstructing the local repository from the last commit"
    git archive -o docker/${ESPNET_ARCHIVE} HEAD  || exit 1
    cd ${SCRIPTPATH}
    test -r ${ESPNET_ARCHIVE} || exit 1;
    sleep 1

    if [ "${build_base_image}" = true ]; then
        echo "building ESPnet base image with ubuntu:${ubuntu_ver}"
        docker build --build-arg DOCKER_VER=${docker_ver} \
                    --build-arg FROM_TAG=${ubuntu_ver} \
                    --build-arg NUM_BUILD_CORES=${build_cores} \
                    -f prebuilt/runtime/Dockerfile -t espnet/espnet:runtime-local . || exit 1
        sleep 1
    fi

    if [[ ${build_ver} == "cpu" ]]; then
        echo "building ESPnet CPU Image with ubuntu:${ubuntu_ver}"
        docker build --build-arg FROM_TAG=runtime-local  --build-arg ESPNET_ARCHIVE=${ESPNET_ARCHIVE} \
                     -f prebuilt/local/Dockerfile -t espnet/espnet:cpu-local . || exit 1
    elif [[ ${build_ver} == "gpu" ]]; then
        echo "building ESPnet GPU Image with ubuntu:${ubuntu_ver} and cuda:${cuda_ver}"
        if [ "${build_base_image}" = true ] ; then
            docker build -f prebuilt/devel/gpu/${ver}/Dockerfile -t espnet/espnet:cuda${ver}-cudnn7 . || exit 1
        else
            if ! [[ -n $( docker images -q espnet/espnet:cuda-latest)  ]]; then
                docker pull espnet/espnet:cuda-latest
            fi
        fi
        build_args="--build-arg FROM_TAG=cuda${ver}-cudnn7"
        build_args="${build_args} --build-arg CUDA_VER=${ver}"
        build_args="${build_args} --build-arg ESPNET_ARCHIVE=${ESPNET_ARCHIVE}"
        docker build ${build_args} -f prebuilt/local/Dockerfile -t espnet/espnet:gpu-cuda${ver}-cudnn7-u18-local . || exit 1
    else
        echo "Parameter invalid: " ${ver}
    fi

    echo "cleanup."
    test -r ${ESPNET_ARCHIVE} && rm ${ESPNET_ARCHIVE}
}

run_recipe1(){
    ./run.sh --docker-egs an4/asr1 \
                        --docker-cmd run.sh \
                        --docker-gpu ${1} \
                        --verbose 1 \
                        --backend ${2} \
                        --ngpu ${3} \
                        --stage ${4} \
                        --tag train_nodev_${2}_${5} | tee -a ${PWD}/testing_${5}_${2}.log > /dev/null
}

run_recipe2(){
   ./run.sh --docker-egs an4/asr1  \
                    --docker-cmd run.sh \
                    --docker-gpu ${1} \
                    --docker-env "NLTK_DATA=/espnet/egs2/an4/asr1/nltk_data,HOME=/espnet/egs2/an4/asr1" \
                    --is-egs2 \
                    --ngpu ${2} \
                    --stage ${3} \
                    --asr-tag train_nodev_${4} \
                    --lm-tag train_nodev_${4}  | tee -a ${PWD}/testing2_pytorch_${4}.log > /dev/null 
}

testing(){
    echo "Testing docker containers"
    # Test Docker Containers with cpu setup
    run_stage=-1
    for backend in chainer pytorch; do
        if [ -f ../egs/an4/asr1/dump/train_nodev/deltafalse/data.json ]; then 
            run_stage=3
        fi
        if [ ! -f .test_cpu_${backend}.done ]; then
            run_recipe1 -1 ${backend} 0 ${run_stage} "cpu"
            touch .test_cpu_${backend}.done
        fi
    done

    for backend in chainer pytorch; do
        if [ -f ../egs/an4/asr1/dump/train_nodev/deltafalse/data.json ]; then 
            run_stage=3
        fi
        if [ ! -f .test_gpu_${backend}.done ]; then
            run_recipe1 0 ${backend} 1 ${run_stage} "gpu"
            touch .test_gpu_${backend}.done
        fi
    done

    echo "ESPnet egs Done. Press <enter> to continue with ESPnet2 egs"
    read enter
    # Test for espnet2
    run_stage=-1
    # 
    if [ ! -f .test2_cpu_${backend}.done ]; then
        run_recipe2 -1 0 ${run_stage} "cpu"
        touch .test2_cpu_${backend}.done
    fi
    run_stage=6
    if [ ! -f .test2_gpu_${backend}.done ]; then
        run_recipe2 0 1 ${run_stage} "gpu"
        touch .test2_gpu_${backend}.done
    fi
}


push(){
    for tag in runtime-latest cuda-latest cpu-latest gpu-latest;do
        echo "docker push espnet/espnet:${tag}"
        ( docker push espnet/espnet:${tag} )|| exit 1
    done
}

## Parameter initialization:
while test $# -gt 0
do
    case "$1" in
        -h) cmd_usage
            exit 0;;
        --help) cmd_usage
            exit 0;;
        --*) ext=${1#--}
            ext=${ext//-/_}
            frombreak=true
            for i in _ {a..z} {A..Z}; do
                for var in `eval echo "\\${!${i}@}"`; do
                    if [ "$var" == "$ext" ]; then
                        eval ${ext}=$2
                        frombreak=false
                        shift
                        break 2
                    fi 
                done 
            done
            if ${frombreak} ; then
                echo "bad option $1" 
                exit 1
            fi
            ;;
        *) break
            ;;
    esac
    shift
done


mode=$1
default_ubuntu_ver=20.04
default_cuda_ver=11.1

check=true
[ "${default_ubuntu_ver}" != "${ubuntu_ver}" ] || [ "${default_cuda_ver}" != "${cuda_ver}" ] && check=false

if [ ${check} = false ] && [ "${mode}" != "fully_local" ]; then
    echo "Error: Use of custom versions of Ubuntu (!=${default_ubuntu_ver}) and CUDA (!=${default_cuda_ver})
        is only available for <mode> == fully_local.
        Exiting... "
    exit 0;
fi


echo "Using Docker Ver.${docker_ver}"
## Application menu
if   [[ "${mode}" == "build" ]]; then
    build
elif [[ "${mode}" == "local" ]]; then
    build_base_image=false
    build_local
elif [[ "${mode}" == "fully_local" ]]; then
    build_base_image=true
    build_local
elif [[ "${mode}" == "push" ]]; then
    push
elif [[ "${mode}" == "test" ]]; then
    testing
elif [[ "${mode}" == "build_and_push" ]]; then
    build
    testing
    push
else
    cmd_usage
fi

echo "$(basename "$0") done."
