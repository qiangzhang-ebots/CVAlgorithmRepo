
---
## How to run algorithm in this repo.


### 1. **Install**

For example, if I want to run the test of `ArithmeticAlgorithmSample`, I can use the following command.

```
colcon build --packages-select ArithmeticAlgorithmSample
```

If you want to debug the test, you should: 

```
colcon build --packages-select ArithmeticAlgorithmSample --cmake-args -DCMAKE_BUILD_TYPE=Debug
```

### 2. **Run**

Add environment variable: `source install/setup.bash`

After the installation is complete, you can run the test file directly. 

```
./build/ArithmeticAlgorithmSample/ArithmeticAlgorithmSample_Test
```

### 3. **Debug**

Add environment variable: `source install/setup.bash`

If you want to debug the test, you need to run the following command.

- create a launch.json file in the .vscode folder.
    - the `launch.json` file should be like this:
    ```
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
            {
                "name": "(gdb) 启动",
                "type": "cppdbg",
                "request": "launch",
                "program": "/bin/bash",
                "args": ["-c","source ${workspaceFolder}/install/setup.bash && ${workspaceFolder}/build/ArithmeticAlgorithmSample/ArithmeticAlgorithmSample_Test"],
                "stopAtEntry": false,
                "cwd": "${fileDirname}",
                "environment": [],
                "externalConsole": false,
                "MIMode": "gdb",
                "setupCommands": [
                    {
                        "description": "为 gdb 启用整齐打印",
                        "text": "-enable-pretty-printing",
                        "ignoreFailures": true
                    },
                    {
                        "description": "将反汇编风格设置为 Intel",
                        "text": "-gdb-set disassembly-flavor intel",
                        "ignoreFailures": true
                    }
                ]
            }

        ]
    }
    ```

    please note that:
    ```
    "program": "/bin/bash",
    "args": ["-c","source ${workspaceFolder}/install/setup.bash && ${workspaceFolder}/build/ArithmeticAlgorithmSample/ArithmeticAlgorithmSample_Test"],
            
    ```
- `F5` to start debug.
