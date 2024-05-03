#include <libgen.h>
#include <unistd.h>

#include "maintest.h"
#include "modulerunner.h"
#include "xgboostparser.h"

using namespace mlir;

namespace Treehierarchy
{
    namespace test
    {    
        // const std::string modelNames[] = {"abalone", "airline", "airline-ohe", "covtype", "epsilon", "letters", "higgs", "year_prediction_msd"};
        const std::string modelNames[] = {"abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "year_prediction_msd"};
        const int32_t NUM_RUNS = 500;

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

        static void verifyXGBoostResult(JsonParser &parser, std::string testCsvPath, std::string answerCsvPath)
        {
            parser.ConstructForest();

            ModuleOp module = parser.buildHIRModule();
            module = parser.lowerToLLVMModule();

            // module->dump();

            Execute::ModuleRunner runner(module);

            utils::CSVReader testCaseReader(testCsvPath);
            std::vector<std::vector<float>> inputData;

            for (size_t i = 0; i < testCaseReader.GetRowNum(); i++)
            {
                auto row = testCaseReader.GetRowOfType<float>(i);
                inputData.push_back(row);
            }

            utils::CSVReader answerReader(answerCsvPath);
            auto epsilon = answerReader.GetRowOfType<float>(0)[0];
            auto answers = answerReader.GetRowOfType<float>(1);

            int i = 0;
            auto classNum = parser.getForestClassNum();
            std::vector<float> result(classNum, -1);

            for (auto &input : inputData) 
            {
                runner.runInference(input.data(), result.data());
                if(classNum > 1) {
                    auto a = std::distance(result.begin(), std::max_element(result.begin(), result.end()));
                    assert(answers[i] == a);
                } else {
                    assert(FPEqual(answers[i], result[0], epsilon));
                }   
                i++;
            }
        }

        static double RunXGBoostTest(JsonParser &parser, std::string testCsvPath)
        {
            parser.ConstructForest();

            ModuleOp module = parser.buildHIRModule();
            module = parser.lowerToLLVMModule();

            module->dump();

            Execute::ModuleRunner runner(module);

            utils::CSVReader reader(testCsvPath);
            std::vector<std::vector<float>> inputData;

            for (size_t i = 0; i < reader.GetRowNum(); i++)
            {
                auto row = reader.GetRowOfType<float>(i);
                inputData.push_back(row);
            }

            std::vector<float> result(parser.getForestClassNum(), -1);

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            for (int32_t trial = 0; trial < NUM_RUNS; trial++) // 500
            {
                for (auto &input : inputData) // 2000
                {
                    runner.runInference(input.data(), result.data());
                }
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            int64_t timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            int64_t numSamples = NUM_RUNS * inputData.size();
            auto timePerSample = (double)timeTaken / (double)numSamples;

            return timePerSample;
        }

        void RunXGBoostNonOptimizeTests()
        {
            std::puts("Running Without Optimize...\n");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                XGBoostParser parser(modelJsonPath, option);

                std::cout << modelName << "time consuming: " << RunXGBoostTest(parser, testCsvPath) << "\n";
            }
        }

        void RunXGBoostSwapOptimizeTests()
        {
            std::puts("Running Swap Optimize...\n");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto stateCsvPath = testModelsDir + "/" + modelName + ".prob.csv";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_swap = true;
                XGBoostParser parser(modelJsonPath, option, stateCsvPath);
                std::cout << modelName << " time consuming: " << RunXGBoostTest(parser, testCsvPath) << "\n";
            }
        }

        void RunXGBoostFlintOptimizeTests()
        {
            std::puts("Running Flint Optimize...\n");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_flint = true;
                XGBoostParser parser(modelJsonPath, option);

                std::cout << modelName << " time consuming: " << RunXGBoostTest(parser, testCsvPath) << "\n";
            }
        }

        void RunXGBoostOptimizeTests()
        {
            std::puts("Running Optimize...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto stateCsvPath = testModelsDir + "/" + modelName + ".prob.csv";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";

                BuildOptions option;
                option.enable_swap = true;
                option.enable_flint = true;
                option.enable_ra = true;
                XGBoostParser parser(modelJsonPath, option, stateCsvPath);

                std::cout << modelName << " time consuming: " << RunXGBoostTest(parser, testCsvPath) << "\n";
            }
        }

        void RunXGBoostCorrectnessTests()
        {
            std::puts("XGBoost Correctness Test...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto stateCsvPath = testModelsDir + "/" + modelName + ".prob.csv";
                auto testCsvPath = testModelsDir + "/" + modelName + ".test.csv";
                auto answerCsvPath = testModelsDir + "/" + modelName + ".answer.csv";

                BuildOptions option;
                option.enable_swap = false;
                option.enable_flint = false;
                option.enable_ra = false;
                option.regNum = 32;
                XGBoostParser parser(modelJsonPath, option, stateCsvPath);

                verifyXGBoostResult(parser, testCsvPath, answerCsvPath);
                std::cout << "Testing model: " << modelName << " success\n";
            }
        }

        void DumpXGBoostLLVMIR()
        {
            std::puts("Dumping XGBoost LLVMIR...");
            for (auto modelName : modelNames)
            {
                auto testModelsDir = GetRepoPath() + "/data/xgb_models";
                auto modelJsonPath = testModelsDir + "/" + modelName + ".json";
                auto stateCsvPath = testModelsDir + "/" + modelName + ".prob.csv";
                auto dumpFileName =  GetRepoPath() + "/xgll/" + modelName + ".ll";

                BuildOptions option;
                option.enable_swap = false;
                option.enable_flint = false;
                option.enable_ra = false;
                option.regNum = 32;

                XGBoostParser parser(modelJsonPath, option, stateCsvPath);
                parser.ConstructForest();
                ModuleOp module = parser.buildHIRModule();
                module = parser.lowerToLLVMModule();

                DumpLLVMIRToFile(module, dumpFileName);
            }
        }
    }
}