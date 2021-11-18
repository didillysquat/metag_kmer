
import os
import re
import pandas as pd
import pickle
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

class JellyCount:   
    def __init__(self):
        self.svd_color_dict = {"SVD1":"purple", "SVD2": "green", "SVD3":"red", "SVD4":"yellow", "SVD5":"blue"}
        if not os.path.exists("meta_info_df.p"):
            with open("svd_poc.txt", "r") as f:
                lines = [_.rstrip() for _ in f]
                self.svd_dict = {_.split("\t")[0]: _.split("\t")[2] for _ in lines}
            self.meta_info_dict = {}

            with open("meta_info.txt", "r") as f:
                lines = [_.rstrip() for _ in f]
            for line in lines:
                try:
                    sample = re.findall("TARA_CO-[0-9]+", line)[0]
                except IndexError:
                    sample = re.findall("TARA_FH-[0-9]+", line)[0]
                if sample not in self.meta_info_dict:
                    site = re.findall("(?<=SITE)[0-9]+", line)[0]
                    island = re.findall("(?<=ISLAND)[0-9]+", line)[0]
                    individual = re.findall("(?<=INDIVIDUAL)[0-9]+", line)[0]
                    svd = self.svd_dict[f"I{island}S{site}C{individual}POC"]
                    self.meta_info_dict[sample] = {"island": island, "site": site, "individual": individual, "svd": svd}
            self.meta_info_df = pd.DataFrame.from_dict(self.meta_info_dict, orient="index")
            with open("meta_info_df.p", "wb") as f:
                pickle.dump(self.meta_info_df, f)
            
        else:
            with open("meta_info_df.p", "rb") as f:
                self.meta_info_df = pickle.load(f)
        

        if os.path.exists("master_kmer_dict_poc.p"):
            
            with open("master_kmer_dict_poc.p", "rb") as f:
                self.master_kmer_dict = pickle.load(f)
            self._get_create_normalised_df()

        else:
            dump_files = [_ for _ in os.listdir('.') if _.endswith("dump.fa")]
            kmer_dicts_dict = {}
            all_kmers = set()
            for dump_file in dump_files:
                name = re.findall("TARA_CO-[0-9]+", dump_file)[0]
                # read in and make a dict of kmer to abundance
                with open(dump_file, "r") as f:
                    lines = [_.rstrip() for _ in f]

                kmer_dict = {lines[i+1]: int(lines[i][1:]) for i in range(0, len(lines), 2) if int(lines[i][1:]) >= 10}
                
                # Keep track of the total number of kmers
                all_kmers.update(set(kmer_dict.keys()))

                kmer_dicts_dict[name] = kmer_dict
            
            self.master_kmer_dict = pd.DataFrame.from_dict(kmer_dicts_dict, orient="index").fillna(0)

            # # Now we should be able to create a df from this dictionary
            # master_kmer_dict = pd.DataFrame.from_dict(new_dicts, orient="index")
            with open("master_kmer_dict_poc.p", "wb") as f:
                pickle.dump(self.master_kmer_dict, f)
    
    def _get_create_normalised_df(self):
        if os.path.exists("normalised_df.p"):
            with open("normalised_df.p", "rb") as f:
                self.normalised_df = pickle.load(f)
                
        else:
            self.normalised_df = self.master_kmer_dict.div(self.master_kmer_dict.sum(axis=1), axis=0)
            with open("normalised_df.p", "wb") as f:
                pickle.dump(self.normalised_df, f)
        self.df_var_sorted =  self.normalised_df.loc[:,self.normalised_df.var(axis=0).sort_values(ascending=False).index]

    def do_LDA(self):
        # Try working with the top n kmers
        
        self.normalised_df_20 =  self.df_var_sorted.iloc[:,:20]
        self.normalised_df_50 =  self.df_var_sorted.iloc[:,:50]
        # Out put the 50 most variable kmers to work with in another method
        with open("top_50_var_kmers.p", "wb") as f:
            pickle.dump(self.normalised_df_50.columns.values, f)
        self.normalised_df_100 =  self.df_var_sorted.iloc[:,:100]
        self.normalised_df_1000 =  self.df_var_sorted.iloc[:,:1000]
        self.normalised_df_5000 =  self.df_var_sorted.iloc[:,:5000]
        
        fig, ax_arr = plt.subplots(ncols=3, nrows=5, figsize=(10,10))
        self._compute_lda(df=self.normalised_df_20, ax_arr=ax_arr[0])
        self._compute_lda(df=self.normalised_df_50, ax_arr=ax_arr[1])
        self._compute_lda(df=self.normalised_df_100, ax_arr=ax_arr[2])
        self._compute_lda(df=self.normalised_df_1000, ax_arr=ax_arr[3])
        self._compute_lda(df=self.normalised_df_5000, ax_arr=ax_arr[4])
        plt.tight_layout()
        plt.savefig("LDA_POC.png", dpi=600)
    
    def do_PCA(self):
        self.normalised_df =  self.df_var_sorted.iloc[:,:1000]
        # Let's try PCA
        x = self.normalised_df.values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        pca = PCA()
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents, columns = [f"PC{_+1}" for _ in range(len(principalComponents))], index=self.normalised_df.index)

        fig, ax_arr = plt.subplots(ncols=3, nrows=1, figsize=(15,5))

        colours = [self.svd_color_dict[self.meta_info_df.at[_,"svd"]] for _ in principalDf.index]
        ax_arr[0].scatter(principalDf["PC1"], principalDf["PC2"], c=colours)
        ax_arr[1].scatter(principalDf["PC1"], principalDf["PC3"], c=colours)
        ax_arr[2].scatter(principalDf["PC1"], principalDf["PC4"], c=colours)

        plt.savefig("PCA_poc.png", dpi=600)

    def do_PCoA(self):
        fig, ax_arr = plt.subplots(ncols=3, nrows=1, figsize=(15,5))
        bc_dm = beta_diversity("braycurtis", self.master_kmer_dict, ids=self.master_kmer_dict.index.values)
        bc_pcoa = pcoa(bc_dm)
        colours = [self.svd_color_dict[self.meta_info_df.at[_,"svd"]] for _ in bc_pcoa.samples.index]
        ax_arr[0].scatter(bc_pcoa.samples["PC1"], bc_pcoa.samples["PC2"],c=colours)
        ax_arr[1].scatter(bc_pcoa.samples["PC1"], bc_pcoa.samples["PC3"],c=colours)
        ax_arr[2].scatter(bc_pcoa.samples["PC1"], bc_pcoa.samples["PC4"],c=colours)
      

    def _compute_lda(self, df, ax_arr):
        # Let's try linear discriminant analysis
        clf = LinearDiscriminantAnalysis()
        x = df.values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        svds = self.meta_info_df.loc[df.index,"svd"].values
        lda = clf.fit_transform(x, svds)
        
        svd_color_dict = {"SVD1":"purple", "SVD2": "green", "SVD3":"red", "SVD4":"yellow", "SVD5":"blue"}
        ax_arr[0].scatter(lda[:,0], lda[:,1], c=[svd_color_dict[_] for _ in svds])
        ax_arr[0].set_xlabel("PC1")
        ax_arr[0].set_ylabel("PC2")
        ax_arr[1].scatter(lda[:,0], lda[:,2], c=[svd_color_dict[_] for _ in svds])
        ax_arr[1].set_xlabel("PC1")
        ax_arr[1].set_ylabel("PC3")
        ax_arr[1].set_title(f"Using top {df.shape[1]} most variable (across samples) kmers")
        ax_arr[2].scatter(lda[:,0], lda[:,3], c=[svd_color_dict[_] for _ in svds])
        ax_arr[2].set_xlabel("PC1")
        ax_arr[2].set_ylabel("PC4")
        
    def investigate_LDA_kmers(self, top_kmer_n=50):
        # The LDA results are pretty amazing. They show us that there is the information in the stacks to differentiate between the groups
        # However, this is supervised so not so great for us.
        # We need to use this data to work out which the good kmers are and which the bad kmers are and then look
        # to see which characters they have in common. I.e is it the abundant kmers that we're after or is it the less abundant kmers
        # I think it wil be interesting to look to see if it is the abundant kmers that are found in one group but not the others.
        # Obviously we'd need to frame this in a nonsupervised way, so we'll be looking to discar all kmers that are not found
        # e.g. in the top 50 most abundant kmers of any sample. Something like that.

        # Let's start by working with the 50 most variable kmers and try to identify their character

        self.normalised_top_var_df =  self.df_var_sorted.iloc[:,:top_kmer_n]

        # We know that variance is a good filter to apply to get us meaningful kmers
        # Let's try to see if we can find other characters that we can filter by
        # Fist look at highest relative abundance vs lowest relative abundance
        # and see where the good kmers lie
        fig, ax = plt.subplots(ncols=1, nrows=1)
        top_kmers = self.normalised_top_var_df.columns.values
        x_max_vals_non_top = [v for k, v in self.normalised_df.max(axis=0).items() if k not in top_kmers]
        x_max_vals_top = [v for k, v in self.normalised_df.max(axis=0).items() if k  in top_kmers]
        x_min_vals_non_top = [v for k, v in self.normalised_df.min(axis=0).items() if k not in top_kmers]
        x_min_vals_top = [v for k, v in self.normalised_df.min(axis=0).items() if k  in top_kmers]
        ax.scatter(x_max_vals_non_top, x_min_vals_non_top, color="black", s=2, alpha=0.1, zorder=1)
        ax.scatter(x_max_vals_top, x_min_vals_top, color="red", s=4, alpha=1, zorder=2)
        ax.set_xlabel("Maximum relative abundance of kmer across all samples")
        ax.set_ylabel("Minimum relative abundance of kmer across all samples")
        plt.tight_layout()
        plt.savefig(f"investigate top {top_kmer_n} variant kmer abundances")
        plt.close()

        # We can see that it is indeed the kmers that are found in high abundance in samples but 0 abundance in others
        # That are coming up as the high variance kmers, and I guess this makes perfect sense.

        # Couple of things to go from here, the number 0.0004 seems like a good upper cutoff for the minimum max abundance
        # and 0 is good for the minimum abundance.
        # I want to know what this 0.0004 cutoff acutaly translates to interms of how abundant the most abundant kmers are.
        print(f"quantile across samples normalised abundance {self.normalised_df.quantile(axis=1, q=0.99)}")
        # It turns out that 0.0004 actually equates to kmers that are extremely abundant (within the top 1%)
        # i.e. .99 quartile so we are only considering extremely abundant kmers.
        # I'd like to know for each sample how many we're left with if we filter by abve quantil .99
        # and that they must be abscent in other samples
        # Let's try that here
        # First work out those kmers that are abscent in at least 1 of the samples
        # We want to work out the quantiles only from the kmers with counts
        quantiles_99_dict = {}
        for sample in self.normalised_df.index:
            ser = self.normalised_df.loc[sample]
            ser = ser[ser > 0]
            quantiles_99_dict[sample] = ser.quantile(.98)
        quantiles_99 = pd.Series(quantiles_99_dict)
        # quantiles_99 = self.normalised_df.quantile(axis=1, q=0.99).values
        selected = self.normalised_df.ge(quantiles_99, axis=0).sum(axis=1)
        counts = (self.normalised_df > 0).sum(axis=1)
        props_kept = selected/counts


        # I also want to look at the kmers of the top 50 that best correlate to the PC that have been found by the LDA
        # and see if we can tease out some characters that way
        clf = LinearDiscriminantAnalysis()
        x = self.normalised_top_var_df.values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        svds = self.meta_info_df.loc[self.normalised_top_var_df.index,"svd"].values
        lda_w_transform = clf.fit_transform(x, svds)
        fig, axarr = plt.subplots(ncols=4, nrows=1, figsize=(15,5))
        # I want to plot up a couple of historams of the scaling
        axarr[0].hist(abs(clf.scalings_[:,0]), bins=100)
        axarr[1].hist(abs(clf.scalings_[:,1]), bins=100)
        axarr[2].hist(abs(clf.scalings_[:,2]), bins=100)
        axarr[3].hist(abs(clf.scalings_[:,3]), bins=100)
        # We want to inspect the scalings
        plt.savefig("LDA_sclaling_hists.png", dpi=600)
        plt.close()

        # This is super cool. It shows that there are only a very few kmers that have strong scalings for the LDA component
        # Let's see which the Kmers are and see if we can recreate the LDA ordiation using these kmers, then
        # we can plot them up on the scatrer to find their properties

        # There is 1 kmer for each of the PCs that has a large scaling
        # I will isolate these 4 kmers
        best_scaling_kmers = []
        for i in range(4):
            ind = np.where(clf.scalings_[:,i] == np.amax(clf.scalings_[:,i]))
            best_scaling_kmers.append(self.normalised_top_var_df.columns.values[ind][0])

        two_kmers = list(set(best_scaling_kmers))
        # Let's see if we can plot up just the two kmers as x and y and see where we get to
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))
        ax.scatter(self.normalised_df[two_kmers[0]], self.normalised_df[two_kmers[1]], c=[self.svd_color_dict[self.meta_info_df.at[_,'svd']] for _ in self.normalised_df.index])

        # best_scaling_kmers = set()
        # for i in range(4):
        #     scaling = pd.Series(abs(clf.scalings_[:,i]), index=self.normalised_top_var_df.columns)
        #     best_scaling_kmers.update(scaling[scaling > 0.15].index.values)
        # foo = "bar"
        
        foo = "bar"

        # Don't forget to check to see at what depth we'll need to be sequencing to pickup the meaningful kmers
        plt.close()

        foo = "bar"

JellyCount().investigate_LDA_kmers()