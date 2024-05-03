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

    protected:
        void CreatePredictFunction() override;
        void CreateLeafNode(Value result, DecisionTree::Node node) override;
        void buildRANodeOp(mlir::Block*, Treehierarchy::DecisionTree*, int);

        arith::CmpFPredicate getComparePredicate() override { return arith::CmpFPredicate::OLE; }
        arith::CmpFPredicate getReverseComparePredicate() override { return arith::CmpFPredicate::OGT; }
        LLVM::ICmpPredicate getCompareIntPredicate() override { return LLVM::ICmpPredicate::sle; }
        LLVM::ICmpPredicate getReverseCompareIntPredicate() override { return LLVM::ICmpPredicate::sgt; }

        FunctionType getTreeFunctionType() override {             
            Type argType = getFeaturePointerType();
            Type resultType = getResultPointerType();
            return m_builder.getFunctionType({argType, resultType}, std::nullopt);
        };

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
                id = m_decisionTree->NewNode(-1.0, DecisionTree::LEAF_NODE_FEATURE, currentNode.prob);
                m_forest->SetClassNum(vec.size());
                m_decisionTree->SetResult(id, vec);
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

    void SklearnParser::buildRANodeOp(Block *entryBlock, DecisionTree *tree, int idx)
    {
        DecisionTree::Node node = tree->GetNode(idx);
        auto loc = m_builder.getUnknownLoc();

        if (!node.IsLeaf())
        {
            Value threshold = createThreshold(node.threshold);
            Value feature;

            int nodeGlobalIdx = m_forest->GetGlobalIdxFromFeature(node.featureIndex);
            //printf("Feature: %d, Get Idx: %d\n", node.featureIndex, nodeGlobalIdx);
            if(m_option.enable_ra && nodeGlobalIdx >= 0)
            {
                feature = pin_reg[nodeGlobalIdx];
            }
            else
            {
                Value featureIdx = m_builder.create<arith::ConstantIntOp>(loc, node.featureIndex, getI32());
                Value input = entryBlock->getArgument(0);
                Value featurePtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getFeatureType(), input, featureIdx);
                feature = m_builder.create<LLVM::LoadOp>(loc, getFeatureType(), featurePtr);
            }
            
            if (m_option.enable_flint && node.threshold < 0)
            {
                Value mask = m_builder.create<arith::ConstantIntOp>(loc, 0x1 << 31, getI32());
                feature = m_builder.create<mlir::arith::XOrIOp>(loc, feature, mask);
            }

            OpBuilder::InsertPoint insertPoint = m_builder.saveInsertionPoint();

            auto leftNode = tree->GetNode(node.leftChild);
            auto rightNode = tree->GetNode(node.rightChild);
            auto predicate = getComparePredicate();
            auto predicate2 = getCompareIntPredicate();
            int64_t leftIdx = node.leftChild;
            int64_t rightIdx = node.rightChild;

            if (m_option.enable_swap && leftNode.probability < rightNode.probability)
            {
                predicate = getReverseComparePredicate();
                predicate2 = getReverseCompareIntPredicate();
                leftIdx = node.rightChild;
                rightIdx = node.leftChild;
            }

            Value condition;
            if (m_option.enable_flint && node.threshold < 0)
            {
                condition = m_builder.create<LLVM::ICmpOp>(loc, predicate2, threshold, feature);
            }
            else if (m_option.enable_flint)
            {
                condition = m_builder.create<LLVM::ICmpOp>(loc, predicate2, feature, threshold);
            }
            else
            {
                condition = m_builder.create<arith::CmpFOp>(loc, predicate, feature, threshold);
            }

            Region *funcBody = entryBlock->getParent();
            Block *tBlock = m_builder.createBlock(funcBody);
            m_builder.setInsertionPointToStart(tBlock);
            buildRANodeOp(entryBlock, tree, leftIdx);

            Block *fBlock = m_builder.createBlock(funcBody);
            m_builder.setInsertionPointToStart(fBlock);
            buildRANodeOp(entryBlock, tree, rightIdx);

            m_builder.restoreInsertionPoint(insertPoint);
            ValueRange nullList = {};
            m_builder.create<LLVM::CondBrOp>(loc, condition, tBlock, nullList, fBlock, nullList);
        }
        else
        {
            Value result = entryBlock->getArgument(1);
            CreateLeafNode(result, node);
        }
    }

    void SklearnParser::CreatePredictFunction()
    {
        Location loc = m_builder.getUnknownLoc();

        Type argType = getFeaturePointerType();
        auto functionType = m_builder.getFunctionType({argType, argType}, std::nullopt);
        auto mainFun = m_builder.create<func::FuncOp>(loc, "predict", functionType);
        mainFun.setPublic();

        Block *callerBlock = mainFun.addEntryBlock();
        m_builder.setInsertionPointToStart(callerBlock);

        // If ra, create tree here
        if(m_option.enable_ra)
        {
            // Get features
            std::vector<size_t> pin_features = m_forest->GetTopFeature();
            size_t regNum = m_forest->GetRegNum();
            for(size_t i = 0; i < regNum; i++)
            {
                // Load data
                Value featureIdx = m_builder.create<arith::ConstantIntOp>(loc, pin_features[i], getI32());
                Value featurePtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getFeatureType(), callerBlock->getArgument(0), featureIdx);
                pin_reg[i] = m_builder.create<LLVM::LoadOp>(loc, getFeatureType(), featurePtr);
            }
            Block *curBlock = m_builder.getBlock();
            // Create tree
            for (size_t i = 0; i < m_forest->GetTreeSize(); i++)
            {
                // Get tree
                DecisionTree *tree = m_forest->GetTree(i);
                buildRANodeOp(curBlock, tree, 0);
                // Coacate next root to previous leaf
                Block *nextBlock = m_builder.createBlock(&mainFun.getBody());
                for(Block* lb: leafBlock)
                {
                    m_builder.setInsertionPointToEnd(lb);
                    m_builder.create<LLVM::BrOp>(loc, nextBlock);
                }
                leafBlock.clear();
                m_builder.setInsertionPointToStart(nextBlock);
            }
        }
        // Else, create function call
        else
        {
            for (size_t i = 0; i < m_forest->GetTreeSize(); i++)
            {
                SmallVector<Value> operands = {callerBlock->getArgument(0), callerBlock->getArgument(1)};
                m_builder.create<func::CallOp>(loc, StringRef("tree_" + std::to_string(i)), getF32(), operands);
            }
        }
        
        m_builder.create<func::ReturnOp>(loc);
        m_module.push_back(mainFun);
    }

    void SklearnParser::CreateLeafNode(Value result, DecisionTree::Node node) 
    {
        auto loc = m_builder.getUnknownLoc();      
        for (size_t i = 0; i < node.result.size(); i++)
        {
            if(node.result[i] != 0.0) {
                Value resultIdx = m_builder.create<arith::ConstantIntOp>(loc, i, getI32());
                Value resultPtr = m_builder.create<LLVM::GEPOp>(loc, getResultPointerType(), getF32(), result, resultIdx);
                Value loadVal = m_builder.create<LLVM::LoadOp>(loc, getF32(), resultPtr);
                Value resultVal = m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(node.result[i]));
                Value addResult = m_builder.create<arith::AddFOp>(loc, resultVal, loadVal);
                m_builder.create<LLVM::StoreOp>(loc, addResult, resultPtr);
            }
        }
        if(m_option.enable_ra)
            leafBlock.push_back(m_builder.getBlock());
        else
            m_builder.create<func::ReturnOp>(loc);
    }

}


#endif