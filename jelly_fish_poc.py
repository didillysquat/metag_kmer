
import os
import re
import pandas as pd
import pickle
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
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
    
    def kmer_mapping(self):
        """
        Write out all of the kmers with their ranks
        Then run mmseqs against our 195 contig to map them to the contig
        THen read this back in and plot up the info
        """

        # Write out the kmers
        if not os.path.exists("poc_kmer_mapping/var_ranked_kmers_poc.fa0"):
            with open("var_ranked_kmers_poc.fa", "w") as f:
                for i, kmer in enumerate(list(self.df_var_sorted)):
                    f.write(f">{i}\n")
                    f.write(f"{kmer}\n")
        
        # The boundaries are as follow (it is inverse)
        # 28S = < 386
        # ITS2 = 386-764
        # 5.8s = 764 - 908
        # ITS1 = 908-1329
        # 18S = > 1329

        # Now for each for each of the kmers plot a line that spans the region to which it maps
        mapps = [[],[]]
        with open("poc_kmer_mapping/var_ranked_on_195.results.sam", "r") as f:
            var_ranked_results = [_.rstrip() for _ in f]
        
        for line in var_ranked_results[2:]:
            comp = line.split("\t")
            if comp[1] == '16' or comp[1] == '0':
                # revcomp mapped
                mapps[0].append(comp[0])
                mapps[1].append(comp[3])
            elif comp[1] == '4':
                # not mapped
                continue
            else:
                foo = "bar"
            foo = "bar"
        x = [int(_) for _ in mapps[0]]
        y = [int(_) for _ in mapps[1]]
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15,5))
        ax.scatter(x, y, zorder=2, s=2, alpha=0.5)
        ax.set_xlim(0, x[-1])
        ax.hlines([386,764,908,1329], xmin=0, xmax=x[-1], colors="black")
        ax.text(s="28S", y=386/2, x=x[-1]/2, ha="center", va="center", fontweight="bold", fontsize=16)
        ax.text(s="ITS2", y=386+((764-386)/2), x=x[-1]/2, ha="center", va="center", fontweight="bold", fontsize=16)
        ax.text(s="5.8S", y=764+((908-764)/2), x=x[-1]/2, ha="center", va="center", fontweight="bold", fontsize=16)
        ax.text(s="ITS1", y=908+((1329-908)/2), x=x[-1]/2, ha="center", va="center", fontweight="bold", fontsize=16)
        ax.text(s="18S", y=1329+((ax.get_ylim()[1]-1329)/2), x=x[-1]/2, ha="center", va="center", fontweight="bold", fontsize=16)
        ax.set_xlabel("kmers ranked by variance; 0 is highest")
        ax.set_ylabel("start of mapping position along the rDNA gene (bp)")
        ax.set_title("kmer mapping sorted by variance rank")
        plt.tight_layout()
        plt.savefig("kmer_mapping_sorted_by_var_rank.png", dpi=600)
        foo = "bar"


    def do_PCA(self, top_kmer_n=250):
        self.normalised_df =  self.df_var_sorted.iloc[:,:top_kmer_n]
        # Let's try PCA
        x = self.normalised_df.values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        pca = PCA()
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents, columns = [f"PC{_+1}" for _ in range(len(principalComponents))], index=self.normalised_df.index)

        fig, ax_arr = plt.subplots(ncols=3, nrows=1, figsize=(15,5))

        colours = [self.svd_color_dict[self.meta_info_df.at[_,"svd"]] for _ in principalDf.index]
        explained = [100*(_/pca.explained_variance.sum()) for _ in pca.explained_variance_]
        ax_arr[0].scatter(principalDf["PC1"], principalDf["PC2"], c=colours)
        ax_arr[0].set_xlabel(f"PC1: {explained[0]:.2f}%")
        ax_arr[0].set_ylabel(f"PC2: {explained[1]:.2f}%")
        
        ax_arr[1].scatter(principalDf["PC1"], principalDf["PC3"], c=colours)
        ax_arr[1].set_xlabel(f"PC1: {explained[0]:.2f}%")
        ax_arr[1].set_ylabel(f"PC3: {explained[2]:.2f}%")
        ax_arr[1].set_title(f"PCA of top {top_kmer_n} highest variance kmers")
        
        ax_arr[2].scatter(principalDf["PC1"], principalDf["PC4"], c=colours)
        ax_arr[2].set_xlabel(f"PC1: {explained[0]:.2f}%")
        ax_arr[2].set_ylabel(f"PC4: {explained[3]:.2f}%")
        
        plt.savefig(f"PCA_poc_{top_kmer_n}.png", dpi=600)

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
        
    def investigate_LDA_kmers(self, top_kmer_n=250):
        # The LDA results are pretty amazing. They show us that there is the information in the stacks to differentiate between the groups
        # However, this is supervised so not so great for us.
        # We need to use this data to work out which the good kmers are and which the bad kmers are and then look
        # to see which characters they have in common. I.e is it the abundant kmers that we're after or is it the less abundant kmers
        # I think it wil be interesting to look to see if it is the abundant kmers that are found in one group but not the others.
        # Obviously we'd need to frame this in a nonsupervised way, so we'll be looking to discar all kmers that are not found
        # e.g. in the top 50 most abundant kmers of any sample. Something like that.

        # Let's start by working with the 50 most variable kmers and try to identify their character
        fig, ax = plt.subplots(ncols=1, nrows=1)
        var_ser = self.df_var_sorted.var(axis=0)
        hist_results = ax.hist(var_ser.values, bins=100)
        rectangles = hist_results[2]
        rect_list = []
        rectangles = sorted(rectangles, key=lambda x: x._x0, reverse=True)
        cumulative = 0
        new_rects = []
        ax.set_ylim(0,50)
        num_kmers = len(self.df_var_sorted.columns)
        max_var = var_ser.max()
        var_cutoff = max_var * 0.75
        for i in range(len(rectangles)):
            if rectangles[i]._x0 > var_cutoff:
                rectangles[i]._set_facecolor((1,0,0,1))
                # new_rect._facecolor = (255,0,0,1)
                # new_rects.append(new_rect)
            # else:
            #     new_rects.append(rect)
            # cumulative += rectangles[i]._height
        plt.savefig(f"POC_variance_hists.png", dpi=600)
        plt.close()

        self.normalised_top_var_df =  self.df_var_sorted.iloc[:,:top_kmer_n]

        # We know that variance is a good filter to apply to get us meaningful kmers
        # Let's try to see if we can find other characters that we can filter by
        # Fist look at highest relative abundance vs lowest relative abundance
        # and see where the good kmers lie
        fig, ax = plt.subplots(ncols=1, nrows=1)
        # top_kmers = self.normalised_top_var_df.columns.values
        top_kmers = ['GATGGGGCCGGGACGCGCCCG', 'GACTGCCGTGCTAGCTGACTT']
        x_max_vals_non_top = [v for k, v in self.normalised_df.max(axis=0).items() if k not in top_kmers]
        x_max_vals_top = [v for k, v in self.normalised_df.max(axis=0).items() if k  in top_kmers]
        x_min_vals_non_top = [v for k, v in self.normalised_df.min(axis=0).items() if k not in top_kmers]
        x_min_vals_top = [v for k, v in self.normalised_df.min(axis=0).items() if k  in top_kmers]
        ax.scatter(x_max_vals_non_top, x_min_vals_non_top, color="black", s=2, alpha=0.01, zorder=1)
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
        plt.close()
        # best_scaling_kmers = set()
        # for i in range(4):
        #     scaling = pd.Series(abs(clf.scalings_[:,i]), index=self.normalised_top_var_df.columns)
        #     best_scaling_kmers.update(scaling[scaling > 0.15].index.values)
        # foo = "bar"
        
        foo = "bar"

        # Let's investigate the informative kmers from the LDA further.
        # Let's pull out the informative and the on informative
        # Then let's look a their variance across samples and plot that up
        # as mean and S.D.

        # A dict that holds the scaling values when working with the 50 most variable kmers
        # that provides the value that, above which we classify the kmer as a 'good' and below a 'bad'
        # for each of the scalings
        if top_kmer_n == 50:
            good_class_dict = {0:100, 1:50, 2:40, 3:30}
        elif top_kmer_n == 250:
            good_class_dict = {0:2, 1:1, 2:0.5, 3:0.4}
        
        good_kmers = set()
        top_kmers = set()
        for k, v in good_class_dict.items():
            ser = pd.Series(abs(clf.scalings_[:,k]), index=self.normalised_top_var_df.columns)
            good_kmers.update(ser[ser > v].index.values)
            top_kmers.add(ser.index.values[np.where(ser.values == ser.max())[0]][0])
        good_kmers = list(good_kmers)
        bad_kmers = [_ for _ in self.normalised_top_var_df.columns if _ not in good_kmers]
        # Write out the kmer list so that we can map it to the ITS boundaries to get an idea of where the informative regions are
        with open(f"LDA_driving_kmers_{top_kmer_n}.fa", "w") as f:
            for kmer in good_kmers:
                f.write(f">{kmer}\n")
                f.write(f"{kmer}\n")
        with open(f"LDA_top_kmers_{top_kmer_n}.fa", "w") as f:
            for kmer in top_kmers:
                f.write(f">{kmer}\n")
                f.write(f"{kmer}\n")

        var_df = self.normalised_top_var_df.var(axis=0)
        good_var = pd.Series(var_df[good_kmers], name="good")
        bad_var = pd.Series(var_df[bad_kmers], name="bad")
        # Look to see what the distribution of the good and bad look like
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))
        ax.scatter(x = [1 for _ in range(len(good_var))], y = good_var.values)
        ax.scatter(x = [2 for _ in range(len(bad_var))], y = bad_var.values)
        ax.set_ylim(0,4.3e-8)
        plt.close()
        # Look to see what the PCA looks like with just the good kmers used
        # good_kmer_df = self.normalised_df[good_kmers]
        good_kmer_df = self.normalised_top_var_df
        x = good_kmer_df.values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        pca = PCA()
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents, columns = [f"PC{_+1}" for _ in range(principalComponents.shape[1])], index=good_kmer_df.index)

        # fig, ax_arr = plt.subplots(ncols=3, nrows=1, figsize=(15,5))
        # explained = [_/pca.explained_variance_.sum() for _ in pca.explained_variance_]
        # colours = [self.svd_color_dict[self.meta_info_df.at[_,"svd"]] for _ in principalDf.index]
        # ax_arr[0].scatter(principalDf["PC1"], principalDf["PC2"], c=colours)
        # ax_arr[0].set_xlabel(f"PC1: {explained[0]}")
        # ax_arr[0].set_ylabel(f"PC2: {explained[1]}")
        
        # ax_arr[1].scatter(principalDf["PC1"], principalDf["PC3"], c=colours)
        # ax_arr[1].set_xlabel(f"PC1: {explained[0]}")
        # ax_arr[1].set_ylabel(f"PC3: {explained[2]}")
        
        # ax_arr[2].scatter(principalDf["PC1"], principalDf["PC4"], c=colours)
        # ax_arr[2].set_xlabel(f"PC1: {explained[0]}")
        # ax_arr[2].set_ylabel(f"PC4: {explained[3]}")
        # plt.tight_layout()
        # plt.close()

        
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))
        # # Also want to see what the split is according ot the single magical 21-mer
        # ax.scatter(self.normalised_df[list(top_kmers)[0]], [1 for _ in range(len(self.normalised_df.index))], c=[self.svd_color_dict[self.meta_info_df.at[_,"svd"]] for _ in self.normalised_df.index])
        # plt.close()


        # For each of the kmers we will plot up the variance against the highest ranking in the four scalings
        # We can also plot up the abundance 

        # Also let's look at the distribution of the variance!

        # So we know from all of this that the most predictive kmers do have a significantly higher variation than the less predicive, but
        # there is a large overlap. SO this is not directly a metric that we can use to further highlight the meaningful kmers.

        # The last think to try is to look at the region that the sequences map to.

        foo = "bar"
        # Don't forget to check to see at what depth we'll need to be sequencing to pickup the meaningful kmers
        

        foo = "bar"

JellyCount().kmer_mapping()