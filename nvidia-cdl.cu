/*
 * Copyright (c) 2014 Smilart and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Alexander Komarov alexander07k@gmail.com - implementation.
 *     
 */

#include <stdio.h>
#include <stdlib.h>
#include <nvml.h>
#include "check.h"
#include "check_cuda.cuh"
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#define PROGRAM_VERSION "1.0"

#define REMOVE_ALL_ATTRIBUTES "\e[m"
#define RED_TEXT "\e[91m"
#define GREEN_TEXT "\e[92m"
#define YELLOW_TEXT "\e[93m"
#define BOLD_TEXT "\e[1m"

struct RunningProcessInfo {
    std::string runningProcessPid;
    std::string runningProcessName;
    std::string runningProcessCommand;

    RunningProcessInfo()
        : runningProcessPid("N/A"),
        runningProcessName("N/A"),
        runningProcessCommand("N/A")
    {}
};

struct DeviceInfo {
    std::string pciBusID;
    std::string name;
    unsigned int temp;
    unsigned long long totalMemory;
    unsigned long long usedMemory;
    unsigned long long freeMemory;
    std::string uuid;
    std::vector<RunningProcessInfo> runningProcessesInfo;

    DeviceInfo() 
        : temp(0),
        name("N/A"),
        pciBusID("N/A"),
        totalMemory(0),
        usedMemory(0),
        uuid("N/A")
    {
    }
};

std::string intToString(int i) {
    std::stringstream out;
    out << i;
    return out.str();
}

std::string nvmlErrorToString(nvmlReturn_t result) {
    return std::string(nvmlErrorString(result));
}

std::string getMemorySizeAsColorString(unsigned long long memorySize) {
    if (memorySize > 1024) {
        return std::string(GREEN_TEXT);
    } else if (memorySize > 200) {
        return std::string(YELLOW_TEXT);
    } else {
        return std::string(RED_TEXT);
    }
}

std::string getTemperatureAsColorString(unsigned int temp) {
    if (temp < 90) {
        return std::string(GREEN_TEXT);
    } else {
        return std::string(RED_TEXT);
    }
}

std::string getBoldText(std::string text) {
    return std::string(BOLD_TEXT + text + REMOVE_ALL_ATTRIBUTES);
}

std::string getColorString(std::string text, std::string color) {
    return std::string(color + text + REMOVE_ALL_ATTRIBUTES);
}

std::vector<std::string> getPCIBusIDByCudaLibrary() {
    int deviceCount = 0;
    cudaError_t result = cudaGetDeviceCount(&deviceCount);
    if (result != cudaSuccess) {
        if (result == cudaErrorNoDevice) {
            throw std::string("no CUDA-capable device is detected.");
        } else {
            check(result == cudaSuccess, cudaGetErrorString(result));
        }
    }
    std::vector<std::string> pciBusIDArray;
    for (int i = 0; i < deviceCount; i++) {
        char pciBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        int arrayLength = NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE;
        checkCudaCall(cudaDeviceGetPCIBusId(pciBusId, arrayLength, i));
        pciBusIDArray.push_back(std::string(pciBusId));
    }
    return pciBusIDArray;
}

std::vector<DeviceInfo> getDeviceInfoByNVML(const std::vector<std::string> &pciBusIDArray) {
    nvmlReturn_t result;
    // Initialise the library.
    result = nvmlInit();
    check(NVML_SUCCESS == result, std::string("Failed to initialise: " + nvmlErrorToString(result) + "\n"));

    char version[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE];
    unsigned int length = NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE;
    result = nvmlSystemGetDriverVersion(version, length);
    std::string driverVersionString = "N/A";
    if (NVML_SUCCESS == result) {
        driverVersionString = std::string(version);
    }
    std::cout << "Driver version: " << getBoldText(driverVersionString) << std::endl;

    // Iterate through the devices.
    std::vector<DeviceInfo> deviceInfoArray;
    for(int i = 0; i < pciBusIDArray.size(); i++) {
        DeviceInfo deviceInfo;

        // Get the device's handle.
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByPciBusId(pciBusIDArray[i].c_str(), &device);
        if (NVML_SUCCESS != result) {
            std::cout << "Failed to get handle for device with this PCI bus ID \"" << pciBusIDArray[i] << "\": " << nvmlErrorToString(result) << std::endl;
            continue;
        }

        // Get the device's name.
        char deviceName[NVML_DEVICE_NAME_BUFFER_SIZE];
        result = nvmlDeviceGetName(device, deviceName, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS == result) {
            deviceInfo.name = std::string(deviceName);
        } else {
            std::cout << "Failed to get device name: " << nvmlErrorToString(result) << std::endl;
        }

        deviceInfo.pciBusID = pciBusIDArray[i];

        // Get the device's temperature.
        unsigned int temp;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        if (NVML_SUCCESS == result) {
            deviceInfo.temp = temp;
        } else {
            std::cout << "Failed to get temperature: " << nvmlErrorToString(result) << std::endl;
        }

        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (NVML_SUCCESS == result) {
            deviceInfo.totalMemory = memory.total / 1024 / 1024;
            deviceInfo.usedMemory = memory.used / 1024 / 1024;
            deviceInfo.freeMemory = memory.free / 1024 / 1024;
        } else {
            std::cout << "Failed to get info of memory: " << nvmlErrorToString(result) << std::endl;
        }

        // Get the device's uuid.
        char deviceUuid[NVML_DEVICE_UUID_BUFFER_SIZE];
        result = nvmlDeviceGetUUID(device, deviceUuid, NVML_DEVICE_UUID_BUFFER_SIZE);
        if (NVML_SUCCESS == result) {
            deviceInfo.uuid = std::string(deviceUuid);
        } else {
            std::cout << "Failed to get device uuid: " << nvmlErrorToString(result) << std::endl;
        }
        deviceInfoArray.push_back(deviceInfo);
    }

    // Shutdown the library.
    result = nvmlShutdown();
    if (NVML_SUCCESS != result) {
        std::cout << "Failed to shutdown: " << nvmlErrorToString(result) << std::endl;
    }

    return deviceInfoArray;
}

void printDeviceInfo(const std::vector<DeviceInfo> &deviceInfo, bool isPrintOptionalInfoOfDevice) {
    for (int i = 0; i < deviceInfo.size(); i++) {
        std::cout << "------------------- " << getBoldText("Device " + intToString(i))  <<" -------------------\n";
        std::cout << "Name: " << getBoldText(deviceInfo[i].name) << std::endl;
        
        std::cout << "Memory usage: ";
        std::cout << std::setw(5) << deviceInfo[i].usedMemory << "MiB" << " / " << std::setw(5) << deviceInfo[i].totalMemory << "MiB" << std::endl;
        std::cout << "Temperature: " << getTemperatureAsColorString(deviceInfo[i].temp) << deviceInfo[i].temp << "C" << REMOVE_ALL_ATTRIBUTES << std::endl;
        if (isPrintOptionalInfoOfDevice) {
            std::cout << "PCI bus ID: " << deviceInfo[i].pciBusID << std::endl;
            std::cout << "UUID: " << deviceInfo[i].uuid << std::endl;
        }

        if (deviceInfo[i].runningProcessesInfo.size() > 0) {
            std::cout << "Running processes on the device:" << std::endl;
            if (isPrintOptionalInfoOfDevice) {
                std::cout << "  PID " << "Command " << std::endl;
                for (int j = 0; j < deviceInfo[i].runningProcessesInfo.size(); ++j) {
                    RunningProcessInfo runningProcessInfo = deviceInfo[i].runningProcessesInfo[j];
                    std::cout.width(5);
                    std::cout << runningProcessInfo.runningProcessPid << " " << runningProcessInfo.runningProcessCommand;
                }
            } else {
                std::cout << "  PID " << "Process name " << std::endl;
                for (int j = 0; j < deviceInfo[i].runningProcessesInfo.size(); ++j) {
                    RunningProcessInfo runningProcessInfo = deviceInfo[i].runningProcessesInfo[j];
                    std::cout.width(5);
                    std::cout << runningProcessInfo.runningProcessPid << " " << runningProcessInfo.runningProcessName << std::endl;
                }
            }
        } else {
            std::cout << getColorString("Running processes not found", RED_TEXT) << std::endl;
        }
        std::cout << std::endl;
    }
}

void printHelpInfo() {
    std::cout << "NVIDIA CUDA device list -- v" << PROGRAM_VERSION << std::endl;
    std::cout << std::endl;
    std::cout << "Gathers device information and compiles a report with enumerated" << std::endl;
    std::cout << "devices in an order consistent with CUDA API." << std::endl;
    std::cout << "Original nvidia-smi does not provide them in any particular order and" << std::endl;
    std::cout << "this can be a problem on machines with multiple GPUs." << std::endl;
    std::cout << std::endl;
    std::cout << "nvidia-cdl [OPTION]" << std::endl << std::endl;
    std::cout << "Options: " << std::endl;
    std::cout << "-h,   --help                Print usage information and exit." << std::endl;
    std::cout << "<no arguments>              Show a summary of GPUs connected to the system." << std::endl;
    std::cout << "-a                          Display extra info." << std::endl;
}

void parseParameters(
    const std::vector<std::string> &argvString,
    bool &isPrintOptionalInfoOfDevice,
    bool &isPrintHelpInfo
) {
    const int argc = argvString.size();
    // the first args is name of file.
    if (argc > 1) {
        bool isAction = false;
        // the first iteration - looking for doesn't provided options.
        for (int i = 1; i < argc; ++i) {
            if (argvString[i].compare("-a") != 0 && argvString[i].compare("-h") != 0 && argvString[i].compare("--help") != 0) {
                throw std::string("Invalid combination of input arguments. Please run 'nvidia-smilart -h' for help.");
            } else {
                // while provides one options at one time.
                if (isAction) {
                    throw std::string("Invalid combination of input arguments. Please run 'nvidia-smilart -h' for help.");
                } else {
                    isAction = true;
                }

            }
        }
        // We have have that there is one provided option.
        // the second iteration - runs executions of certain commands.
        for (int i = 1; i < argc; ++i) {
            if (argvString[i].compare("-a") == 0) {
                isPrintOptionalInfoOfDevice = true;
            } else if (argvString[i].compare("-h") == 0 || argvString[i].compare("--help") == 0) {
                isPrintHelpInfo = true;
            } else {
                throw std::string("Invalid combination of input arguments. Please run 'nvidia-smilart -h' for help.");
            }
        }
    }
}

void findProccessesRunningOnDevices(std::vector<DeviceInfo> &deviceInfoFromNVML) {
    // ps -A --format pid,command | grep "\-\-gpu=0"
    std::string command("ps -A --format pid,command | grep \"\\-\\-gpu=\"");

    for (int i = 0; i < deviceInfoFromNVML.size(); ++i) {
        std::string customCommand(command + intToString(i));
        FILE *file = popen(customCommand.c_str(), "r");

        if (file == NULL) {
            std::cout << "Failed to find processes running on devices: opening pipe failed" << std::endl;
            continue;
        }

        char line[1000];
        while (fgets(line, 1000, file) != NULL) {
            std::string lineString = std::string(line);
            // delete gap in begin of string.
            while (lineString.size() > 0 && lineString.compare(0, 1, " ") == 0) {
                lineString = lineString.substr(1);
            }

            RunningProcessInfo runningProcessInfo;

            // getting pid
            int pidStringEndIndex = lineString.find(" ");
            if (pidStringEndIndex == -1) {
                std::cout << "Failed to find processes running on devices: internal error" << std::endl;
                continue;
            }
            std::string pidString = lineString.substr(0, pidStringEndIndex);

            // getting name
            int nameStringEndIndex = lineString.find(" ", pidStringEndIndex + 1);
            if (nameStringEndIndex == -1) {
                std::cout << "Failed to find processes running on devices: internal error" << std::endl;
                continue;
            }
            std::string fullProcessNameString = lineString.substr(pidStringEndIndex + 1, nameStringEndIndex - pidStringEndIndex -1);
            int onlyNameStringStartIndex = fullProcessNameString.find_last_of("/");
            if (onlyNameStringStartIndex == -1) {
                std::cout << "Failed to find processes running on devices: internal error" << std::endl;
                continue;
            }
            std::string processNameString = fullProcessNameString.substr(onlyNameStringStartIndex + 1, fullProcessNameString.size() - onlyNameStringStartIndex);

            std::string processCommandString = lineString.substr(pidStringEndIndex + 1, lineString.size() - pidStringEndIndex - 1);

            runningProcessInfo.runningProcessPid = pidString;
            runningProcessInfo.runningProcessCommand = processCommandString;
            runningProcessInfo.runningProcessName = processNameString;

            deviceInfoFromNVML[i].runningProcessesInfo.push_back(runningProcessInfo);
        }
        
        pclose(file);
    }
}

int main(int argc, char** argv) {
    try {
        // option -a
        bool isPrintOptionalInfoOfDevice = false;
        bool isPrintHelpInfo = false;

        std::vector<std::string> argvString;
        for (int i = 0; i < argc; ++i) {
            argvString.push_back(std::string(argv[i]));
        }
        parseParameters(argvString, isPrintOptionalInfoOfDevice, isPrintHelpInfo);

        if (isPrintHelpInfo) {
            printHelpInfo();
            return 0;
        }

        const std::vector<std::string> pciBusIDArray = getPCIBusIDByCudaLibrary();
        std::vector<DeviceInfo> deviceInfoFromNVML = getDeviceInfoByNVML(pciBusIDArray);

        findProccessesRunningOnDevices(deviceInfoFromNVML);

        printDeviceInfo(deviceInfoFromNVML, isPrintOptionalInfoOfDevice);
        
    } catch (std::string &message) {
        std::cout << message << std::endl;
    } catch (...) {
        std::cout << "Error occured." << std::endl;
        std::cout << "Use the standard nvidia-smi." << std::endl;
    }
    return 0;
}
