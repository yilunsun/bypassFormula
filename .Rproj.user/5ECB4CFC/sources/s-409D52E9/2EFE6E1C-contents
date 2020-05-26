#include <RcppArmadillo.h>
#include <sstream>
#include <numeric>
#include "Random.h"
#include "Model.h"
#include "Observation.h"
#include "Density.h"
#include "NodeTree.h"
using namespace Rcpp;
using namespace std;
using namespace arma;
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]

/*** R
npCBPS_neo <- function(y,x) {
  library(CBPS)
  #print("X:", dim(X))
  return(1/(length(y)*unlist(npCBPS(y~x)['weights'])))
}

gam_neo <- function(a,ps,y,a_out) {
  library(mgcv)
  fit <- gam(y~s(a)+s(ps,by=a))
  pred <- predict(fit, data.frame(ps=ps,a=rep(a_out,each=length(ps))))
  optvalue <- apply(matrix(pred, nrow = length(ps)), 2, mean)
  return(optvalue)
}
*/

NodeTree::NodeTree()
{
	parent=NULL;
	this->nLeaf=0;
	this->nLevel=0;
	this->splitVariable=-1;
	left=NULL;
	right=NULL;
	return;
};

NodeTree::NodeTree(const Observation *obs)
{
	this->SetObservation(obs);
	int n=this->GetObservation()->GetN();
	this->SubjectList.resize(n);
	for (int i=0;i<n;i++)
		SubjectList[i]=i; //Initially, SubjectList has all the observations in the dataset.*/
	parent=NULL;
	left=NULL;
	right=NULL;
	splitVariable=-1;
	for (int i = 0; i < obs->GetP(); i++)
	  Splitting_var.push_back(i);
	this->nLeaf=0;
	this->nLevel=0;
	this->temperature=obs->GetTemp();
	return;
};

NodeTree::~NodeTree()
{
	if (left!=NULL)
	{
		left->RemoveSubTree();
		delete left;
		left=NULL;
	};
	if (right!=NULL)
	{
		right->RemoveSubTree();
		delete right;
		right=NULL;
	};
	return;
};

NodeTree::NodeTree(Node &root)
{
	parent=NULL;
	this->nLeaf=0;
	this->nLevel=0;
	Node* NewNode=root.CopySubTree();
	this->SetLeftNode(NewNode->GetLeftNode());
	this->SetRightNode(NewNode->GetRightNode());
	this->SetObservation(NewNode->GetObservation());
	this->SetTemperature(NewNode->GetTemperature());
	this->SetSplitLevel(NewNode->GetSplitLevel());
	this->SetSplitVariable(NewNode->GetSplitVariable());
	this->SetSubjectList(NewNode->GetSubjectList());
	this->SetSplittingVar(NewNode->GetSplittingVar());
	this->SetMinimumLeafSize(NewNode->GetMinimumLeafSize());
	this->SetMinLeaf(NewNode->GetMinLeaf());
	this->SetAllMinLeaf(NewNode->GetMinLeaf());
	return;
}

void NodeTree::SetParentNode(Node *node)
{
	Rcout<<"The parent for a tree can only be NULL"<<endl;
};

NodeTree* NodeTree::CopyTree(void) const
{
	NodeTree* NewTree=new NodeTree;
	NewTree->SetObservation(this->GetObservation());
	NewTree->SetSplitLevel(this->GetSplitLevel());
	NewTree->SetSplitVariable(this->GetSplitVariable());
	NewTree->SetSplittingVar(this->GetSplittingVar());
	NewTree->SetTemperature(this->GetTemperature());
	NewTree->SetSubjectList(this->GetSubjectList());
	NewTree->SetMinimumLeafSize(this->GetMinimumLeafSize());
	NewTree->SetMinLeaf(this->GetMinLeaf());
	if (this->GetLeftNode()!=NULL || this->GetRightNode()!=NULL)
	{
		Node* pLeft=this->GetLeftNode()->CopySubTree();
		Node* pRight=this->GetRightNode()->CopySubTree();
		NewTree->SetLeftNode(pLeft);
		NewTree->SetRightNode(pRight);
	}

	return NewTree;
};

double NodeTree::Pot(const Model &modelStructure, const Model &modelVar,  const Model &modelLikelihood, Random &ran)
{
	double pot=0;
	pot =  modelStructure.Potential(*this,ran);
	pot += modelVar.Potential(*this, ran);
	pot += modelLikelihood.Potential(*this, ran);
	potential=pot;
	return pot;
};

double NodeTree::LogLikelihood(const Model &modelLikelihood, Random &ran)
{
	double pot=0;
	pot += modelLikelihood.Potential(*this, ran);
	return -pot;
};

int NodeTree::GetSize() const
{
	NodeTree *Tree=this->CopyTree();
  std::vector <Node *> node;
	node.push_back(Tree);
	std::vector <Node *> leaves;
	int i;
	int Size;

	for (i=0;i<node.size();i++)
	{
		if (node[i]->GetRightNode()==NULL)
		{
			leaves.push_back(node[i]);
		}
		else
		{

			node.push_back(node[i]->GetLeftNode());
			node.push_back(node[i]->GetRightNode());
		};
	};
	Size=leaves.size();
	delete Tree;

	return Size;
};

int NodeTree::GetMiniNodeSize() const
{
  NodeTree *Tree=this->CopyTree();
  std::vector <Node *> node;
  node.push_back(Tree);
  std::vector <Node *> leaves;
  
  for (int i=0;i<node.size();i++)
  {
    if (node[i]->GetRightNode()==NULL)
    {
      leaves.push_back(node[i]);
    }
    else
    {
      node.push_back(node[i]->GetLeftNode());
      node.push_back(node[i]->GetRightNode());
    };
  };
  
  int MiniNodeSize = 100000;
  
  for (int i=0;i<leaves.size();i++) //loop through leaves
  {
    std::vector <int> SubjectList=leaves[i]->GetSubjectList();
    if (MiniNodeSize >= SubjectList.size()) {
      MiniNodeSize = SubjectList.size();
    };
  };
  
  
  delete Tree;
  
  return MiniNodeSize;
};

std::vector <Node *> NodeTree::GetLeaves()
{
	NodeTree *Tree=this->CopyTree();
  std::vector <Node *> node;
	node.push_back(Tree);
	std::vector <Node *> leaves;
	int i;

	for (i=0;i<node.size();i++)
	{
		if (node[i]->GetRightNode()==NULL)
		{
			leaves.push_back(node[i]);
		}
		else
		{

			node.push_back(node[i]->GetLeftNode());
			node.push_back(node[i]->GetRightNode());
		};
	};
	return leaves;
};

NumericVector NodeTree::GCBPS(const NumericMatrix& x, const NumericVector& y){ // this implementation requires expose predict in xgboost package as predict2
  
  // Obtain environment containing function
  Rcpp::Environment package_env("package:bypassFormula"); 
  
  Rcpp::Function npCBPS_neo = package_env["npCBPS_neo"];  
  
  // Call the function and receive output (might not be list)
  //Rcout << "here";
  NumericVector rf_obj = npCBPS_neo(Named("y")=y, _["x"]=x);
  //Rcout << "2";
  
  return rf_obj;
}

std::vector<double> NodeTree::GAM(const NumericVector& a, const NumericVector& ps, const NumericVector& y, const NumericVector& a_out){ 
  
  // Obtain environment containing function
  Rcpp::Environment package_env("package:bypassFormula"); 
  
  Rcpp::Function gam_neo = package_env["gam_neo"];  
  
  // Call the function and receive output (might not be list)
  //Rcout << "here";
  NumericVector a_pred = gam_neo(Named("a")=a, _["ps"]=ps, _["y"]=y, _["a_out"]=a_out);
  //Rcout << "2";
  
  return arma::conv_to<std::vector<double> >::from(a_pred);
}

List NodeTree::GetVal() //get value of a tree and optimal label for each node
{
  double value = 0; //value is the value of the tree
  std::vector <int> label; //label is optimal dose vector for each leaf
  
  NodeTree *Tree=this->CopyTree();
  std::vector <Node *> node;
  node.push_back(Tree);
  std::vector <Node *> leaves;
  int i;
  
  for (i=0;i<node.size();i++)
  {
    if (node[i]->GetRightNode()==NULL)
    {
      leaves.push_back(node[i]);
    }
    else
    {
      node.push_back(node[i]->GetLeftNode());
      node.push_back(node[i]->GetRightNode());
    };
  };
  
  for (i=0;i<leaves.size();i++) //loop through leaves
  {
    std::vector <int> SubjectList=leaves[i]->GetSubjectList();
    
    int opt = 0;
    
    // start retriving data for ith leaf
    NumericVector y_leaf(SubjectList.size()), a_leaf(SubjectList.size()), a_eval = obs->GetCanDose();
    NumericMatrix x_leaf(SubjectList.size(), obs->GetP());
    
    for (int j=0; j<SubjectList.size(); j++){ //loop within leaf node to get all observed data in that leaf
      y_leaf(j) = obs->GetV(SubjectList[j]);
      x_leaf(j,_) = obs->GetXi(SubjectList[j]);
      a_leaf(j) = obs->GetA(SubjectList[j]); // observed dose level vector
    };

    //Rcout<<"x_leaf max: "<<max(x_leaf)<<"\n";
    //Rcout<<"values: "<<value<<"\t";
    NumericVector gps_leaf = GCBPS(x_leaf, a_leaf);
    std::vector<double> est = GAM(a_leaf, gps_leaf, y_leaf, a_eval); //vector of estimated reward
    double maxval;
    vector_max(est, maxval, opt);
    value = value + maxval;
    if (std::isnan(value)) {
      List temp_list = List::create(Named("y_leaf")=y_leaf,
                               Named("a_leaf")=a_leaf,
                               Named("x_leaf")=x_leaf);
      return List::create(Named("label")=temp_list,
                          Named("value")=-999,
                          Named("fail")=true);
    }
      
    label.push_back(opt);
  };
  
  delete Tree;
  leaves.clear();
  node.clear();
  
  return List::create(Named("label")=label,
                      Named("value")=value,
                      Named("fail")=false);
};

std::vector <std::vector <double> > NodeTree::GetLeavesObservations()
{
  std::vector <std::vector <double> > Result;
	//vector <Node *> leaves=GetLeaves();
	//int i;

	NodeTree *Tree=this->CopyTree();
	std::vector <Node *> node;
	node.push_back(Tree);
	std::vector <Node *> leaves;
	int i;

	for (i=0;i<node.size();i++)
	{
		if (node[i]->GetRightNode()==NULL)
		{
			leaves.push_back(node[i]);
		}
		else
		{

			node.push_back(node[i]->GetLeftNode());
			node.push_back(node[i]->GetRightNode());
		};
	};

	for (i=0;i<leaves.size();i++)
	{
	  std::vector <double> temp;
	  std::vector <int> SubjectList=leaves[i]->GetSubjectList();
		for (int j=0;j<SubjectList.size();j++)
			temp.push_back(obs->GetY(SubjectList[j]));
		Result.push_back( temp );
	};
	delete Tree;
	leaves.clear();
	node.clear();


	return Result;
};

std::vector <std::vector <int> > NodeTree::GetLeavesSubjects()
{
  std::vector <std::vector <int> > Result;
	//vector <Node *> leaves=GetLeaves();
	//int i;

	NodeTree *Tree=this->CopyTree();
	std::vector <Node *> node;
	node.push_back(Tree);
	std::vector <Node *> leaves;
	int i;

	for (i=0;i<node.size();i++)
	{
		if (node[i]->GetRightNode()==NULL)
		{
			leaves.push_back(node[i]);
		}
		else
		{

			node.push_back(node[i]->GetLeftNode());
			node.push_back(node[i]->GetRightNode());
		};
	};

	for (i=0;i<leaves.size();i++)
	{
	  std::vector <int> SubjectList=leaves[i]->GetSubjectList();
		Result.push_back(SubjectList);
	};
	delete Tree;
	leaves.clear();
	node.clear();

	return Result;
};

std::vector <int> NodeTree::GetSplittingVariableSet() const
{
  std::vector <int> Result;
	//vector <Node *> leaves=GetLeaves();
	//int i;

	NodeTree *Tree=this->CopyTree();
	std::vector <Node *> node;
	node.push_back(Tree);
	std::vector <Node *> leaves;
	int i;

	for (i=0;i<node.size();i++)
	{
		if (node[i]->GetRightNode()==NULL)
		{
			leaves.push_back(node[i]);
		}
		else
		{
			Result.push_back(node[i]->GetSplitVariable());
			node.push_back(node[i]->GetLeftNode());
			node.push_back(node[i]->GetRightNode());
		};
	};

	delete Tree;
	leaves.clear();
	node.clear();

	return Result;
};

bool NodeTree::Misclassfication(int &LeavesSize, int &Error, Density *like) const
{

	NodeTree *Tree=this->CopyTree();
  std::vector <Node *> node;
	node.push_back(Tree);
	std::vector <Node *> leaves;
	int i;
	int k = Tree->GetObservation()->GetK();

	LeavesSize=0; Error=0;


	for (i=0;i<node.size();i++)
	{
		if (node[i]->GetRightNode()==NULL) //leaf
		{
			leaves.push_back(node[i]);

		  std::vector<int> subject(node[i]->GetSubjectList());
			const Observation *obs = node[i]->GetObservation();
			std::vector <int> s1;

			int l = subject.size();

			int s,j;

			NumericVector leaf;

			for (s = 0; s < l; s++)
			{
			  int nr = subject[s];
			  leaf.push_back(obs->GetY(nr));
			  for (j = 0; j < k; j++){
			    int count = 0;
			    if (obs->GetY(nr)==j) count++;
			    s1.push_back(count);
			  };
			};

			std::vector <double> pMean=like->PosteriorMean(leaf);

			pMean.push_back(1-std::accumulate(pMean.begin(), pMean.end(), 0.0));
			int pmax=std::distance(pMean.begin(), std::max_element(pMean.begin(), pMean.end()));
			if (pmax==k)
			  s1.erase(s1.begin());
			else
			  s1.erase(s1.begin()+pmax-1);

			Error+=std::accumulate(s1.begin(), s1.end(), 0);
		}
		else
		{
			node.push_back(node[i]->GetRightNode());
			node.push_back(node[i]->GetLeftNode());
		};
	};
	LeavesSize=leaves.size();
	delete Tree;
	Tree=NULL;
	return true;
};

// NumericVector NodeTree::seq_length(const double& start, const double& end, const int& length_out){
//   NumericVector result;
//   double increment = (end-start)/(length_out - 1);
//   for (int i = 0; i < length_out; i++){
//     result.push_back(start+i*increment);
//   }
//   return result;
// }

void NodeTree::vector_max(std::vector<double> v, double &max, int &imax){
  std::vector<double>::size_type p=0;
  imax = -1;
  max = std::numeric_limits<double>::lowest();
  
  for (auto &val : v)
  {
    if (!std::isnan(val) && val>max)
    {
      imax = p;
      max = val;
    }
    p++;
  }
}

void NodeTree::checkTree(void) const
{
  NodeTree *Tree=this->CopyTree();
  std::vector <Node *> node;
  node.push_back(Tree);
  std::vector <Node *> leaves;
  
  for (int i=0;i<node.size();i++)
  {
    Rcout<<"Now Checking MinLeaf is: "<<node[i]->GetMinLeaf()<<"\t";
    if (node[i]->GetRightNode()==NULL)
    {
      leaves.push_back(node[i]);
    }
    else
    {
      node.push_back(node[i]->GetLeftNode());
      node.push_back(node[i]->GetRightNode());
    };
  };
  Rcout<<"Finished Checking MinLeaf\n";
  delete Tree;
};

