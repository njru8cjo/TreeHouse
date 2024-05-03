#include <libgen.h>
#include <unistd.h>
#include <iomanip>

#include "csvreader.h"
#include "maintest.h"
#include "modulerunner.h"
#include "sklearnparser.h"

namespace Treehierarchy
{
    namespace test
    {
        const std::string modelNames[] = {"adult", "bank", "letter", "magic", "satlog", "spambase", "wine-quality", "sensorless"};
        const int32_t NUM_RUNS = 10000;

        static std::string GetRepoPath()
        {
            char exePath[PATH_MAX];
            memset(exePath, 0, sizeof(exePath));
            if (readlink("/proc/self/exe", exePath, PATH_MAX) == -1)
                return std::string("");
            char *execDir = dirname(exePath);
            char *buildDir = dirname(execDir);
            char *repoPath = dirname(buildDir);
            return repoPath;
        }

        static double RunSklearnTest(JsonParser &parser, std::string testCsvPath) {
            
            parser.ConstructForest();
            ModuleOp module = parser.buildHIRModule();
            module = parser.lowerToLLVMModule();

            Execute::ModuleRunner runner(module);

            utils::CSVReader reader(testCsvPath);
            std::vector<std::vector<float>> inputData;

            for (size_t i = 0; i < reader.GetRowNum(); i++)
            {
                auto row = reader.GetRowOfType<float>(i);
                row.erase(row.begin());
                inputData.push_back(row);
            }
            
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            for (int32_t trial = 0; trial < NUM_RUNS; trial++)
            {
                for (auto &input : inputData)
                {
                    std::vector<float> result(parser.getForestClassNum(), 0);
                    runner.runInference(input.data(), result.data());
                }
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            int64_t timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            int64_t numSamples = NUM_RUNS * inputData.size();
            auto timePerSample = (double)timeTaken / (double)numSamples;
            return timePerSample;
        }

        static void verifySKlearnResult(JsonParser &parser, std::string testCsvPath, std::string answerCsvPath)
        {
            parser.ConstructForest();
            ModuleOp module = parser.buildHIRModule();
            // module->dump();
            module = parser.lowerToLLVMModule();

            Execute::ModuleRunner runner(module);

            utils::CSVReader testCaseReader(testCsvPath);
            std::vector<std::vector<float>> inputData;

            for (size_t i = 0; i < testCaseReader.GetRowNum(); i++)
            {
                auto row = testCaseReader.GetRowOfType<float>(i);
                row.erase(row.begin());
                inputData.push_back(row);
            }

            utils::CSVReader answerReader(answerCsvPath);
            auto epsilon = answerReader.GetRowOfType<float>(0)[0];
            auto classNum = parser.getForestClassNum();

            for(size_t i = 0; i < inputData.size(); i++) {
                std::vector<float> input = inputData[i];
                std::vector<float> result(classNum, 0);
                std::vector<float> answer = answerReader.GetRowOfType<float>(i+1);
                runner.runInference(input.data(), result.data());
                for(size_t x = 0; x < classNum; x++) {
                   assert(FPEqual(answer[x], result[x], epsilon));
                }
            }
        }

        void RunSKlearnNonOptimizeTests()
        {
            std::puts("Running Without Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                SklearnParser parser(modelJsonPath, option);

                auto time = RunSklearnTest(parser, testCsvPath);
                std::cout << modelName << " time consuming: " << time << "\n\n";
            }           
        }

        void RunSKlearnSwapOptimizeTests()
        {
            std::puts("Running Swap Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_swap = true;
                SklearnParser parser(modelJsonPath, option);

                auto time = RunSklearnTest(parser, testCsvPath);
                std::cout << modelName << " time consuming: " << time << "\n\n";
            }           
        }

        void RunSKlearnFlintOptimizeTests()
        {
            std::puts("Running Flint Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_flint = true;
                SklearnParser parser(modelJsonPath, option);

                auto time = RunSklearnTest(parser, testCsvPath);
                std::cout << modelName << " time consuming: " << time << "\n\n";
            }           
        }

        void RunSKlearnOptimizeTests()
        {
            std::puts("Running Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_swap = true;
                option.enable_flint = true;
                SklearnParser parser(modelJsonPath, option);

                auto time = RunSklearnTest(parser, testCsvPath);
                std::cout << modelName << " time consuming: " << time << "\n";
            }
            std::cout << std::endl;
        }

        void RunSKlearnCorrectnessTests()
        {
            std::puts("SKlearn Correctness Test...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";
                auto answerCsvPath = testModelsDir + "/" + modelName + ".answer.csv";

                BuildOptions option;
                option.enable_swap = false;
                option.enable_flint = false;
                option.enable_ra = false;
                option.regNum = 32;
                SklearnParser parser(modelJsonPath, option);

                verifySKlearnResult(parser, testCsvPath, answerCsvPath);
                std::cout << "Testing model: " << modelName << " success\n";
            }
        }

        void DumpSKlearnLLVMIR()
        {
            std::puts("Dumping SKlearn LLVMIR...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/sklearn_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto dumpFileName =  GetRepoPath() + "/skll/" + modelName + ".ll";

                BuildOptions option;
                option.enable_swap = false;
                option.enable_flint = false;
                option.enable_ra = false;
                option.regNum = 32;
                SklearnParser parser(modelJsonPath, option);
                parser.ConstructForest();
                ModuleOp module = parser.buildHIRModule();
                module = parser.lowerToLLVMModule();

                DumpLLVMIRToFile(module, dumpFileName);
            }
        }
    }
}