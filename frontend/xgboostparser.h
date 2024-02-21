#include "jsonparser.h"

using json = nlohmann::json;

class XGBoostParser : public JsonParser
{
public:
    XGBoostParser(const std::string& treeJSON): JsonParser(treeJSON) {}
    
    ~XGBoostParser() {}
    void constructForest() override;
    void constructTree(const json treeJSON);
};

void XGBoostParser::constructForest()
{
    auto& learnerJSON = m_json["learner"];

    auto& featureTypesJSON = learnerJSON["feature_types"];
    m_forest->setFeatureSize(featureTypesJSON.size());

    auto& boosterJSON = learnerJSON["gradient_booster"];
    auto& modelJSON = boosterJSON["model"];
    auto& treesJSON = modelJSON["trees"];

    for (auto& treeJSON : treesJSON)
    {
        m_decisionTree = &(m_forest->newTree());
        constructTree(treeJSON);
    }
}

void XGBoostParser::constructTree(const json treeJSON) {
 
    auto& left_children = treeJSON["left_children"];
    auto& right_childen = treeJSON["right_children"];
    auto& parents = treeJSON["parents"];
    auto& thresholds = treeJSON["split_conditions"];
    auto& featureIndices = treeJSON["split_indices"];

    auto num_nodes = static_cast<size_t>(std::stoi(treeJSON["tree_param"]["num_nodes"].get<std::string>()));
    std::vector<int64_t> nodeIds;

    for (size_t i=0 ; i< num_nodes ; ++i)
    {
        int64_t nodeId = m_decisionTree->NewNode(thresholds[i].get<double>(), featureIndices[i].get<int64_t>(), 1.0);
        nodeIds.push_back(nodeId);
    }

    for (size_t i=0 ; i< num_nodes ; ++i)
    {
        auto leftChildIndex = left_children[i].get<int>();
        if (leftChildIndex != -1)
            m_decisionTree->SetNodeLeftChild(nodeIds[i], nodeIds[leftChildIndex]);
        auto rightChildIndex = right_childen[i].get<int>();
        if (rightChildIndex != -1)
            m_decisionTree->SetNodeRightChild(nodeIds[i], nodeIds[rightChildIndex]);
        
        auto parentIndex = parents[i].get<int>();
        if (parents[i].get<int>() == 2147483647)
            m_decisionTree->SetNodeParent(nodeIds[i], DecisionTree::ROOT_NODE_PARENT);
        else
            m_decisionTree->SetNodeParent(nodeIds[i], nodeIds[parentIndex]);
    }
}