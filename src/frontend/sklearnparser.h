#ifndef SKLERAN_PARSER_H
#define SKLERAN_PARSER_H

#include "jsonparser.h"
#include <queue>

using json = nlohmann::json;

namespace Treehierarchy
{
    class SklearnParser : public JsonParser
    {
    public:
        SklearnParser(const std::string &forestJSONPath, BuildOptions option) : JsonParser(forestJSONPath, option) {}

        ~SklearnParser() {}

        void ConstructForest() override;
        void ConstructTree(const json treeJSON);

    private:
        void CreatePredictFunction() override;
        struct Node
        {
            json node;
            int64_t parent;
            double prob;
            bool isLeft;
        };
    };

    void SklearnParser::ConstructForest()
    {
        for (auto &treeJSON : m_json)
        {
            m_decisionTree = &(m_forest->newTree());
            ConstructTree(treeJSON);
        }
    }

    void SklearnParser::ConstructTree(const json treeJSON)
    {
        int64_t id;
        std::queue<Node> nodeQueue;
        Node currentNode;

        nodeQueue.push({treeJSON, DecisionTree::ROOT_NODE_PARENT, DecisionTree::ROOT_NODE_PROB, true});

        while (!nodeQueue.empty())
        {
            currentNode = nodeQueue.front();

            if (currentNode.node.value("isCategorical", "True") == "False")
            {
                int32_t featureIndex = currentNode.node["feature"].get<int32_t>();
                double threshold = currentNode.node["split"].get<double>();
                double probLeft = currentNode.node["probLeft"].get<double>();
                double probRight = currentNode.node["probRight"].get<double>();
                id = m_decisionTree->NewNode(threshold, featureIndex, currentNode.prob);

                nodeQueue.push({currentNode.node["leftChild"], id, probLeft * currentNode.prob, true});
                nodeQueue.push({currentNode.node["rightChild"], id, probRight * currentNode.prob, false});
            }
            else
            {
                std::vector<float> vec = currentNode.node["prediction"].get<std::vector<float>>();
                auto maxIt = std::max_element(vec.begin(), vec.end());
                double prediction = std::distance(vec.begin(), maxIt);
                m_forest->SetClassNum(vec.size()); // TODO: better way to get classNum?

                id = m_decisionTree->NewNode(prediction, DecisionTree::LEAF_NODE_FEATURE, currentNode.prob);
            }

            int64_t parentId = currentNode.parent;

            m_decisionTree->SetNodeParent(id, parentId);

            if (currentNode.isLeft && parentId != DecisionTree::ROOT_NODE_PARENT)
            {
                m_decisionTree->SetNodeLeftChild(parentId, id);
            }
            else if (parentId != DecisionTree::ROOT_NODE_PARENT)
            {
                m_decisionTree->SetNodeRightChild(parentId, id);
            }

            nodeQueue.pop();
        }

        // Todo: Maybe a better way to get feature size
        for(auto node: m_decisionTree->GetNodes()) {
            m_forest->SetFeatureProb(std::make_pair(node.featureIndex, node.probability));
            if(m_forest->GetFeatureSize() < node.featureIndex + 1)
                m_forest->SetFeatureSize(node.featureIndex + 1);
        }
        m_forest->SortFeatureProb();  
    }

    // TODO: this should be finish
    void SklearnParser::CreatePredictFunction()
    {
        Location loc = m_builder.getUnknownLoc();

        Type argType = getFeaturePointerType();
        auto functionType = m_builder.getFunctionType({argType, argType}, std::nullopt);
        auto mainFun = m_builder.create<func::FuncOp>(loc, "predict", functionType);
        mainFun.setPublic();

        Block *callerBlock = mainFun.addEntryBlock();
        m_builder.setInsertionPointToStart(callerBlock);

        Value input = callerBlock->getArgument(1);
        auto size = m_forest->GetClassNum();
        
        auto oneFConst = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(1.0));

        for (size_t i = 0; i < m_forest->GetTreeSize(); i++)
        {
            auto callResult = m_builder.create<func::CallOp>(loc, StringRef("tree_" + std::to_string(i)), getF32(), callerBlock->getArgument(0));
            Value idx = m_builder.create<arith::FPToUIOp>(loc, getI32(), callResult.getResult(0));

            Value resultPtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getF32(), input, idx);
            Value frequency = m_builder.create<LLVM::LoadOp>(loc, getFeatureType(), resultPtr);
            mlir::Value addResult = m_builder.create<arith::AddFOp>(loc, oneFConst, frequency);
            m_builder.create<LLVM::StoreOp>(loc, addResult, resultPtr);
        }

        m_builder.create<func::ReturnOp>(loc);
        m_module.push_back(mainFun);
    }
}

#endif